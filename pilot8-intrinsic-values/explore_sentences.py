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


u, b, c = fit(np.ones(len(rows), bool))
u0, _, _ = fit(wd == 0)
u1, _, _ = fit(wd == 1)
p0 = np.array([u[sid[(v, 0)]] for v in names])
p1 = np.array([u[sid[(v, 1)]] for v in names])
out = {
    "b": b, "bias_towards_A": c,
    "same_sentence_across_wordings_r": float(np.corrcoef(u0, u1)[0, 1]),
    "two_sentences_of_a_value_r": float(np.corrcoef(p0, p1)[0, 1]),
    "spread_sd_sentences": float(u.std()),
    "spread_sd_values_mean_of_two": float(((p0 + p1) / 2).std()),
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
