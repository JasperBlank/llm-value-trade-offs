"""Pilot 10 analysis (DESIGN.md sections 3-4), on CPU.

  python analyze10.py [results/real]     reads menu_pA.npy and ratings10.json there; writes report.md,
                                         verdicts10.json and prices10.json there
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
P8 = HERE.parent / "pilot8-intrinsic-values"
WN = ["W0 how good", "W1 how much better the world", "W2 how bad or good", "W3 how much it matters to you"]
F1_MIN, F2_MIN, F3_MIN, RIDGE = 0.8, 0.7, 1.0, 0.01


def r(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def fit_prices(keys, p, n_s, lam=RIDGE):
    """P(A) = sigmoid(u_A - u_B + b log(nA/nB) + c), maximum likelihood on the soft answers, ridge lam * sum(u^2)."""
    a, bb = keys[:, 0], keys[:, 1]
    dlog = np.log(keys[:, 2] / keys[:, 3])
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)

    def f(x):
        u = x[:n_s]
        z = u[a] - u[bb] + x[n_s] * dlog + x[n_s + 1]
        q = np.clip(1 / (1 + np.exp(-z)), 1e-12, 1 - 1e-12)
        g = q - p
        nll = -np.sum(p * np.log(q) + (1 - p) * np.log(1 - q)) + lam * np.sum(u ** 2)
        gu = np.bincount(a, g, n_s) - np.bincount(bb, g, n_s) + 2 * lam * u
        return nll, np.concatenate([gu, [np.sum(g * dlog)], [np.sum(g)]])
    x = minimize(f, np.concatenate([np.zeros(n_s), [1.0, 0.0]]), jac=True, method="L-BFGS-B").x
    u = x[:n_s]
    z = u[a] - u[bb] + x[n_s] * dlog + x[n_s + 1]
    lo = lambda q: np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
    return u - u.mean(), float(x[n_s]), float(x[n_s + 1]), r(z, lo(p))


def main():
    res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real")
    S = json.load(open(HERE / "sentences10.json"))["sentences"]
    n_s = len(S)
    keys = np.array(json.load(open(HERE / "menu_keys10.json")))
    pA = np.load(res / "menu_pA.npy")
    R = json.load(open(res / "ratings10.json"))
    if len(pA) != len(keys):                      # the quick test runs a subset
        keys = keys[:: R.get("keys_menu_subset_every", 1)]
    u, b, c, fit_r = fit_prices(keys, pA, n_s)
    # split-half reliability: unordered pairs split at random, each half fitted on its own
    rng = np.random.default_rng(10)
    pair = np.minimum(keys[:, 0], keys[:, 1]) * n_s + np.maximum(keys[:, 0], keys[:, 1])
    up = np.unique(pair)
    half = set(rng.choice(up, len(up) // 2, replace=False).tolist())
    m = np.array([x in half for x in pair])
    u1, u2 = fit_prices(keys[m], pA[m], n_s)[0], fit_prices(keys[~m], pA[~m], n_s)[0]
    split_r = r(u1, u2)

    lp = np.array(R["logp"], dtype=float)
    lean7 = np.log(np.exp(lp[:, 8:]).sum(1) + 1e-300) - np.log(np.exp(lp[:, :7]).sum(1) + 1e-300)
    RD = {"rating": {}, "score": {}, "lean7": {}}
    for k, x, s_, l7 in zip(R["keys"], R["rating"], R["score"], lean7):
        RD["rating"][tuple(k)], RD["score"][tuple(k)], RD["lean7"][tuple(k)] = x, s_, l7
    sizes = sorted({k[1] for k in RD["rating"]})
    get = lambda kind, w, n, idx=range(n_s): np.array([RD[kind][(i, n, w)] for i in idx])

    # F1: the full range
    f1 = r(get("rating", 0, 8), u)
    # F2: the log-scale score on pilot 9's 48 sentences (4 people) against pilot 8's prices
    ex = json.load(open(P8 / "results/real/exploratory_sentences.json"))
    p8u = {(s["value"], s["phrasing"]): s["u"] for s in ex["sentences"]}
    mid = [i for i, s in enumerate(S) if s["level"] == "medium"]
    u8 = np.array([p8u[(S[i]["value"], S[i]["p8"])] for i in mid])
    f2 = r(get("score", 0, 4, mid), u8)
    # F3: size
    f3 = float(np.mean(get("rating", 0, 32) - get("rating", 0, 2)))
    V = {"F1": {"r": f1, "pass": f1 >= F1_MIN}, "F2": {"r": f2, "pass": f2 >= F2_MIN},
         "F3": {"gain_2_to_32": f3, "size_sensitive": f3 >= F3_MIN},
         "menu": {"b": b, "bias_towards_A": c, "fit_r": fit_r, "split_half_r": split_r}}

    # the rest (DESIGN.md section 4)
    vals = sorted({s["value"] for s in S})
    idx = {v: {lev: [i for i, s in enumerate(S) if s["value"] == v and s["level"] == lev] for lev in ("small", "medium", "large")} for v in vals}
    order_ok = lambda x: sum(max(x[idx[v]["small"]]) < min(x[idx[v]["medium"]]) and max(x[idx[v]["medium"]]) < min(x[idx[v]["large"]]) for v in vals)
    rate8 = get("rating", 0, 8)
    level = np.array([["small", "medium", "large"].index(s["level"]) for s in S])
    vid = np.array([vals.index(s["value"]) for s in S])
    X_level = np.eye(3)[level]
    X_value = np.eye(len(vals))[vid]

    def r2(X, y):
        beta = np.linalg.lstsq(X, y - y.mean(), rcond=None)[0]
        return float(1 - np.sum((y - y.mean() - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2))
    V["designed_order"] = {"menu": order_ok(u), "rating_W0_8": order_ok(rate8), "of": len(vals)}
    V["variance_explained"] = {"menu": {"stakes_level": r2(X_level, u), "value": r2(X_value, u), "both": r2(np.hstack([X_level, X_value]), u)},
                               "rating_W0_8": {"stakes_level": r2(X_level, rate8), "value": r2(X_value, rate8),
                                               "both": r2(np.hstack([X_level, X_value]), rate8)}}
    uv = u[mid]
    V["context"] = {"medium_prices_vs_pilot8_r": r(uv, u8),
                    "pair_same_as_pilot8": int(sum(np.sign(uv[2 * i] - uv[2 * i + 1]) == np.sign(u8[2 * i] - u8[2 * i + 1]) for i in range(len(vals))))}
    tab = {}
    for kind in ("rating", "score", "lean7"):
        for w in range(4):
            x8, x4m = get(kind, w, 8), get(kind, w, 4, mid)
            tab[f"{kind}|{WN[w]}"] = {"all96_r": r(x8, u), "within_level_r": r(x8 - np.array([x8[level == L].mean() for L in level]),
                                                                             u - np.array([u[level == L].mean() for L in level])),
                                      "medium48_new_menu_r": r(get(kind, w, 8, mid), uv), "medium48_pilot8_r": r(x4m, u8),
                                      "pairs_same_as_new_menu": int(sum(np.sign(get(kind, w, 8, mid)[2 * i] - get(kind, w, 8, mid)[2 * i + 1]) == np.sign(uv[2 * i] - uv[2 * i + 1]) for i in range(len(vals))))}
    V["by_readout"] = tab
    gain = {lev: float(np.mean((get("rating", 0, 32) - get("rating", 0, 2))[level == k])) for k, lev in enumerate(("small", "medium", "large"))}
    G = np.array([get("rating", 0, n) for n in sizes])
    slope = np.array([np.polyfit(np.log(sizes), G[:, j], 1)[0] for j in range(n_s)])
    V["size"] = {"gain_by_level": gain, "slope_per_log_person": float(slope.mean()),
                 "slope_in_menu_units": float(slope.mean() * u.std() / rate8.std()), "menu_b": b}
    V["ratings_at_level"] = {lev: {"mean": float(rate8[level == k].mean()), "sd": float(rate8[level == k].std())}
                             for k, lev in enumerate(("small", "medium", "large"))}
    V["share_near_7"] = float(np.mean(np.abs(np.array([RD["rating"][k] for k in RD["rating"] if k[2] == 0]) - 7) < 0.1))
    json.dump(V, open(res / "verdicts10.json", "w"), indent=1, default=float)
    prices = sorted([{"value": s["value"], "level": s["level"], "text": s["text"], "u": float(u[i]), "rating_W0_8": float(rate8[i]),
                      "score_W0_8": float(get("score", 0, 8)[i])} for i, s in enumerate(S)], key=lambda d: -d["u"])
    json.dump(prices, open(res / "prices10.json", "w"), indent=1)

    L = [f"# Pilot 10 results ({R['model']})", "",
         f"Menu: {len(keys)} trade-offs; P(A) = sigmoid(u_A - u_B + b log(nA/nB) + c), b = {b:.2f}, bias towards A {c:+.2f}; "
         f"the fit orders the answers with r = {fit_r:.2f}; split-half reliability of the 96 prices r = {split_r:.2f}. "
         f"Letters take at least {R['letter_mass_min']:.3f} of the answer probability.", "",
         "## Pre-registered tests", "",
         f"- **F1** the full range: 'How good would it be …' (8 people) against the menu price, 96 sentences: r = {f1:.2f}: "
         f"**{'pass' if f1 >= F1_MIN else 'fail'}** (bar {F1_MIN})",
         f"- **F2** the log-scale score on pilot 9's 48 sentences (4 people) against pilot 8's prices: r = {f2:.2f}: "
         f"**{'pass' if f2 >= F2_MIN else 'fail'}** (bar {F2_MIN}; the expected rating gave 0.55)",
         f"- **F3** 32 people against 2: the rating rises by {f3:+.2f} points on average: **{'size-sensitive' if f3 >= F3_MIN else 'not size-sensitive'}** (bar {F3_MIN})", "",
         "## The menu across the range", "",
         f"- Designed order (small < both medium < large) holds for {V['designed_order']['menu']} of 24 values in the menu, "
         f"{V['designed_order']['rating_W0_8']} in the rating.",
         f"- Share of the price spread explained by the stakes level alone: {V['variance_explained']['menu']['stakes_level']:.2f}; "
         f"by the value alone: {V['variance_explained']['menu']['value']:.2f}; both: {V['variance_explained']['menu']['both']:.2f}. "
         f"For the rating: {V['variance_explained']['rating_W0_8']['stakes_level']:.2f}, {V['variance_explained']['rating_W0_8']['value']:.2f}, "
         f"{V['variance_explained']['rating_W0_8']['both']:.2f}.",
         f"- Pilot 8's 48 sentences in this new menu against their pilot 8 prices: r = {V['context']['medium_prices_vs_pilot8_r']:.2f}; "
         f"the same sentence of each pair wins in {V['context']['pair_same_as_pilot8']} of 24.",
         "- Mean W0 rating at 8 people: " + ", ".join(f"{k} {v['mean']:.2f} (SD {v['sd']:.2f})" for k, v in V["ratings_at_level"].items())
         + f"; {V['share_near_7']:.0%} of W0 answers within 0.1 of 7.", "",
         "## By readout and wording", "",
         "| readout | wording | all 96 (8 people) | within stakes level | pilot 8's 48, new menu | pilot 8's 48 (4 people), pilot 8 prices | pairs: same as new menu (of 24) |",
         "|---|---|---|---|---|---|---|"]
    for k, t in tab.items():
        kind, w = k.split("|")
        L.append(f"| {kind} | {w} | {t['all96_r']:.2f} | {t['within_level_r']:.2f} | {t['medium48_new_menu_r']:.2f} | {t['medium48_pilot8_r']:.2f} | {t['pairs_same_as_new_menu']} |")
    L += ["", "## Size (W0)", "",
          "- Gain from 2 to 32 people by level: " + ", ".join(f"{k} {v:+.2f}" for k, v in gain.items()) + ".",
          f"- Per log(people): {V['size']['slope_per_log_person']:.2f} points; in the menu's units {V['size']['slope_in_menu_units']:.2f} against b = {b:.2f}.", "",
          "## The price list (menu price; W0 rating at 8 people)", "", "| # | value | level | sentence | price | rating |", "|---|---|---|---|---|---|"]
    for i, d in enumerate(prices, 1):
        L.append(f"| {i} | {d['value']} | {d['level']} | {d['text']} | {d['u']:+.2f} | {d['rating_W0_8']:.2f} |")
    (res / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
