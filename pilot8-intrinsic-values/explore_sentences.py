"""Exploratory, after the results (not pre-registered): are the prices about the values or about the
sentences? Uses the model's saved answers (results/real/qwen/behaviour_pA.npy, in behaviour.jsonl order).

1. A price for each of the 48 sentences (24 values x 2 phrasings), fitted on all trade-offs.
2. The same, fitted separately on the two question wordings: if a sentence's price is stable across
   wordings but the two sentences of a value disagree, the model prices the sentence, not the value."""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot8 import fit_worth  # noqa: E402

res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real")
rows = [json.loads(l) for l in open(HERE / "behaviour.jsonl")]
pa = np.load(res / "qwen/behaviour_pA.npy").astype(np.float64)
assert len(pa) == len(rows)
vals = json.load(open(HERE / "values.json"))["values"]
names = [v["name"] for v in vals]
text = {(v["name"], i): p for v in vals for i, p in enumerate(v["phrasings"])}
sid = {k: i for i, k in enumerate(itertools.product(names, (0, 1)))}
sa = np.array([sid[(r["a"], r["pa"])] for r in rows])
sb = np.array([sid[(r["b"], r["pb"])] for r in rows])
dlog = np.array([math.log(r["na"] / r["nb"]) for r in rows])
wd = np.array([r["wording"] for r in rows])


def fit(mask):
    u, b, c, _ = fit_worth(sa[mask], sb[mask], dlog[mask], pa[mask], len(sid))
    return u - u.mean(), b, c


def logodds(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def nll_fit(n_par, z_of, x0):
    """Maximum likelihood for P(A) = sigmoid(z_of(x)), z linear in x; gradient by autodiff-free chain rule."""
    from scipy.optimize import minimize

    def f(x):
        z, J = z_of(x)
        q = np.clip(1 / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
        return -np.sum(pa_m * np.log(q) + (1 - pa_m) * np.log(1 - q)), J(q - pa_m)
    return minimize(f, x0, jac=True, method="L-BFGS-B").x


u, b, c = fit(np.ones(len(rows), bool))
u0, b0, c0 = fit(wd == 0)
u1, _, _ = fit(wd == 1)
p0 = np.array([u[sid[(v, 0)]] for v in names])
p1 = np.array([u[sid[(v, 1)]] for v in names])

# How much of the behaviour one price per sentence explains: fitted on wording 0, predicting wording 1's
# answers; the ceiling is the same prompt answered under the two wordings.
z1 = u0[sa[wd == 1]] - u0[sb[wd == 1]] + b0 * dlog[wd == 1] + c0
key = lambda r: (r["a"], r["pa"], r["b"], r["pb"], r["na"], r["nb"])
k0 = {key(r): i for i, r in enumerate(rows) if r["wording"] == 0}
i0, i1 = np.array([(k0[key(r)], i) for i, r in enumerate(rows) if r["wording"] == 1]).T
explained = {"sentence_prices_predict_other_wording_r": float(np.corrcoef(z1, logodds(pa[wd == 1]))[0, 1]),
             "sentence_prices_predict_other_wording_same_side": float(((z1 > 0) == (pa[wd == 1] > 0.5)).mean()),
             "ceiling_same_prompt_two_wordings_r": float(np.corrcoef(logodds(pa[i0]), logodds(pa[i1]))[0, 1]),
             "ceiling_same_prompt_two_wordings_same_side": float(((pa[i0] > 0.5) == (pa[i1] > 0.5)).mean())}

# Order: a separate price for each sentence in slot A and in slot B.
pa_m = pa
nS = len(sid)


def z_order(x):
    uA, uB = x[:nS], x[nS:2 * nS]
    z = uA[sa] - uB[sb] + x[2 * nS] * dlog + x[2 * nS + 1]
    return z, lambda g: np.concatenate([np.bincount(sa, g, nS), -np.bincount(sb, g, nS), [np.sum(g * dlog)], [np.sum(g)]])


xo = nll_fit(2 * nS + 2, z_order, np.zeros(2 * nS + 2))
uA, uB = xo[:nS] - xo[:nS].mean(), xo[nS:2 * nS] - xo[nS:2 * nS].mean()

# Size: a separate weight of log(people) for each sentence: worth(s, n) = u_s + b_s log n.
la = np.array([math.log(r["na"]) for r in rows])
lb = np.array([math.log(r["nb"]) for r in rows])


def fit_bs(m):
    global pa_m
    pa_m = pa[m]
    A, B, LA, LB = sa[m], sb[m], la[m], lb[m]

    def z_size(x):
        uu, bs = x[:nS], x[nS:2 * nS]
        z = uu[A] - uu[B] + bs[A] * LA - bs[B] * LB + x[2 * nS]
        return z, lambda g: np.concatenate([np.bincount(A, g, nS) - np.bincount(B, g, nS),
                                            np.bincount(A, g * LA, nS) - np.bincount(B, g * LB, nS), [np.sum(g)]])
    x = nll_fit(2 * nS + 1, z_size, np.concatenate([np.zeros(nS), np.full(nS, 2.0), [0.0]]))
    pa_m = pa
    return x[nS:2 * nS]


bs_all, bs0, bs1 = fit_bs(np.ones(len(rows), bool)), fit_bs(wd == 0), fit_bs(wd == 1)
inv = {i: k for k, i in sid.items()}
order_size = {"price_slot_A_vs_slot_B_r": float(np.corrcoef(uA, uB)[0, 1]), "price_sd_slot_A": float(uA.std()),
              "price_sd_slot_B": float(uB.std()),
              "size_weight_per_sentence": {"median": float(np.median(bs_all)), "q10": float(np.quantile(bs_all, 0.1)),
                                           "q90": float(np.quantile(bs_all, 0.9)),
                                           "wording0_vs_wording1_r": float(np.corrcoef(bs0, bs1)[0, 1]),
                                           "lowest": [[inv[i][0], text[inv[i]], float(bs_all[i])] for i in np.argsort(bs_all)[:3]],
                                           "highest": [[inv[i][0], text[inv[i]], float(bs_all[i])] for i in np.argsort(bs_all)[-3:][::-1]]}}
out = {
    "b": b, "bias_towards_A": c,
    "same_sentence_across_wordings_r": float(np.corrcoef(u0, u1)[0, 1]),
    "two_sentences_of_a_value_r": float(np.corrcoef(p0, p1)[0, 1]),
    "spread_sd_sentences": float(u.std()),
    "spread_sd_values_mean_of_two": float(((p0 + p1) / 2).std()),
    "explained": explained, "order_and_size": order_size,
    "sentences": sorted([{"value": v, "phrasing": i, "text": text[(v, i)], "u": float(u[sid[(v, i)]]),
                          "u_wording0": float(u0[sid[(v, i)]]), "u_wording1": float(u1[sid[(v, i)]])}
                         for v in names for i in (0, 1)], key=lambda s: -s["u"]),
}
json.dump(out, open(res / "exploratory_sentences.json", "w"), indent=1)

gap = sorted(names, key=lambda v: -abs(p0[names.index(v)] - p1[names.index(v)]))
L = ["# Exploratory: prices of the sentences, not the values", "",
     "Not pre-registered; computed after the results from the model's saved answers (`explore_sentences.py`).", "",
     f"- The same sentence priced from the two question wordings separately: r = {out['same_sentence_across_wordings_r']:.2f}.",
     f"- The two sentences written for the same value: r = {out['two_sentences_of_a_value_r']:.2f}.",
     f"- Spread of the 48 sentence prices: SD {out['spread_sd_sentences']:.2f} logits; b = {b:.2f} per log person.", "",
     "## How far one price per sentence goes", "",
     f"- **It explains most of what is reliable.** Prices fitted on one question wording predict the answers under the other with "
     f"r = {explained['sentence_prices_predict_other_wording_r']:.2f} ({explained['sentence_prices_predict_other_wording_same_side']:.0%} on the same side). "
     f"The ceiling, the same prompt answered under the two wordings, is r = {explained['ceiling_same_prompt_two_wordings_r']:.2f} "
     f"({explained['ceiling_same_prompt_two_wordings_same_side']:.0%}).",
     f"- **Order: mostly.** A sentence's price as option A and as option B agree r = {order_size['price_slot_A_vs_slot_B_r']:.2f}. "
     f"Prices spread a little more as option B (SD {order_size['price_sd_slot_B']:.2f} against {order_size['price_sd_slot_A']:.2f}).",
     "- **Size: the weight of the number of people depends on the sentence.** With a separate weight of log(people) for each "
     f"sentence, the weights have median {order_size['size_weight_per_sentence']['median']:.2f} (10-90%: "
     f"{order_size['size_weight_per_sentence']['q10']:.2f} to {order_size['size_weight_per_sentence']['q90']:.2f}), and they are reliable: "
     f"r = {order_size['size_weight_per_sentence']['wording0_vs_wording1_r']:.2f} between the two wordings. Lowest: "
     + "; ".join(f"'{t}' {w:.2f}" for _, t, w in order_size["size_weight_per_sentence"]["lowest"]) + ". Highest: "
     + "; ".join(f"'{t}' {w:.2f}" for _, t, w in order_size["size_weight_per_sentence"]["highest"]) + ".", "",
     "## The values whose two sentences disagree most", "",
     "| value | sentence | price u |", "|---|---|---|"]
for v in gap[:8]:
    for i in (0, 1):
        L.append(f"| {v if i == 0 else ''} | {text[(v, i)]} | {u[sid[(v, i)]]:+.2f} |")
L += ["", "## All 48 sentences, highest price first", "", "| # | value | sentence | price u |", "|---|---|---|---|"]
for k, s in enumerate(out["sentences"], 1):
    L.append(f"| {k} | {s['value']} | {s['text']} | {s['u']:+.2f} |")
(res / "exploratory_sentences.md").write_text("\n".join(L) + "\n")
print("\n".join(L[:12]))
print("\n".join(L[-48 - 4:][:30]))
