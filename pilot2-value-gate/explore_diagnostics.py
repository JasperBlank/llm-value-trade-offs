"""EXPLORATORY, written after seeing the results (not pre-registered).

Why do the models fail the gate? Prints, per model (test goods, strangers):
- how decisive the answers are, and the letter-position bias;
- how well other phrasings of the same prompt predict it (a ceiling for any model that
  does not know the wording), a separate fit for every pair, and one value per good
  (in sample and on held-out pairs);
- the letter preference in three-option menus;
- for 14B: the goal module in the raw answers, by option order.
Writes results/real/exploratory_diagnostics.md.
"""
import collections

import numpy as np

from common import ROOT, load_jsonl
from fit import UtilityFit, cv_correlation, two_option_table

OUT = ROOT / "results" / "real"
md = ["# Exploratory diagnostics (after the results; not pre-registered)", "",
      "Test goods, strangers. All correlations are over individual prompts.", "",
      "| model | answers near 0 or 1 | mean P(first option) | letter bias (fit) | other phrasings predict a prompt | separate fit per pair | one value per good, in sample | one value per good, held-out (criterion 2) | gains only, held-out | harms only, held-out | 3-option menus: P(A) / P(B) / P(C) |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for m in ["4b", "8b", "14b"]:
    rows = load_jsonl(OUT / f"test_{m}.jsonl")
    T = [r for r in two_option_table(rows) if r["kind"] in ("within", "cross") and r["beneficiary"] == "strangers"]
    p = np.array([r["p"] for r in T])
    by = collections.defaultdict(dict)
    for r in T:
        by[(r["x"], r["qx"], r["y"], r["qy"])][r["phrasing"]] = r["p"]
    pred, obs = [], []
    for v in by.values():
        if len(v) == 3:
            for k in v:
                obs.append(v[k])
                pred.append(np.mean([v[j] for j in v if j != k]))
    ceiling = np.corrcoef(pred, obs)[0, 1]
    pairs = collections.defaultdict(list)
    for r in T:
        pairs[tuple(sorted((r["x"], r["y"])))].append(r)
    pp, oo = [], []
    for recs in pairs.values():
        f = UtilityFit(recs, "log")
        pp += list(f.predict(recs))
        oo += [r["p"] for r in recs]
    f = UtilityFit(T, "log")
    one_in = np.corrcoef(f.predict(T), p)[0, 1]
    within = [r for r in T if r["kind"] == "within"]
    gains = [r for r in within if r["x"].startswith("G_")]
    harms = [r for r in within if r["x"].startswith("H_")]
    menus = [r["probs"] for r in rows if len(r["options"]) == 3 and r["beneficiary"] == "strangers"]
    letters = np.mean(menus, axis=0)
    md.append(f"| {m} | {np.mean((p < 0.01) | (p > 0.99)):.0%} | {p.mean():.2f} | {f.c:+.2f} (quantity slope b = {f.b:.2f}) | "
              f"{ceiling:.2f} | {np.corrcoef(pp, oo)[0, 1]:.2f} | {one_in:.2f} | {cv_correlation(T, 'log')[0]:.2f} | "
              f"{cv_correlation(gains, 'log')[0]:.2f} | {cv_correlation(harms, 'log')[0]:.2f} | "
              f"{letters[0]:.2f} / {letters[1]:.2f} / {letters[2]:.2f} |")

md += ["", "Letter bias: the fitted log-odds added to the first option (negative = prefers the second). "
       "For comparison, 1 against 8 people moves the log-odds by b × ln 8.", "",
       "## Goal module in the raw answers (14B, strangers)", "",
       "P(the re-labelled box is chosen), 1 box against 8 people, by the box's position.", "",
       "| real good | with the box objective: box first | box second | without an objective: box first | box second |",
       "|---|---|---|---|---|"]
G = [r for r in load_jsonl(OUT / "goal_14b.jsonl") if r["beneficiary"] == "strangers"]
N = [r for r in load_jsonl(OUT / "test_14b.jsonl") if r["beneficiary"] == "strangers" and r["kind"] == "rvn"]


def p_good(rows, good, q_good, other, q_other):
    out = {}
    for r in rows:
        if len(r["options"]) != 2:
            continue
        d = {x["good"]: (i, x["q"]) for i, x in enumerate(r["options"])}
        if set(d) == {good, other} and d[good][1] == q_good and d[other][1] == q_other:
            i = d[good][0]
            out.setdefault(i, []).append(r["probs"][i] / sum(r["probs"]))
    return [float(np.mean(out[i])) if i in out else float("nan") for i in (0, 1)]


for real in ["G_vacc", "G_groc", "G_dayoff", "H_wrist", "H_scam", "H_double"]:
    a = p_good([r for r in G if r["kind"] == "goal2"], "N_box", 1, real, 8)
    b = p_good(N, "N_box", 1, real, 8)
    md.append(f"| {real} | {a[0]:.3f} | {a[1]:.3f} | {b[0]:.3f} | {b[1]:.3f} |")
text = "\n".join(md) + "\n"
(OUT / "exploratory_diagnostics.md").write_text(text, encoding="utf-8")
print(text)
