"""Score Claude's and Jasper's pilot-7 predictions against the results. Probabilities: Brier score
(lower is better; multi-way questions summed over their options). Numbers: does the 80% range
contain the result, and the distance of the best guess. Jasper answered a short set; a question
only one of the two answered is scored for that one."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real")
V = json.load(open(res / "verdicts7.json"))
q = V["qwen"]
mp = json.load(open(res / "qwen/map.json"))
das = json.load(open(res / "qwen/das.json"))
T = mp["swaps"]["good"]["transfer"]
Ls = mp["layers"]
v5 = q["b_site"]["V5"]
v5key = ("comparison" if v5.startswith("the comparison") else "shared_code" if v5.startswith("a value code")
         else "not_testable" if v5.startswith("not testable") else "a_own_value")
outcome = {
    "q1_after_A_half_layer": next(L for L in Ls if T["after_A"][str(L)]["value"] >= 0.5),
    "q2_a_site_share": max(T[s][str(L)]["value"] for s in ("A_end", "A_nl") for L in Ls if L <= das["choice"]["arrival_layer"]),
    "q3_b_site": das["choice"]["b_site"][0],
    "q4_a_site_number_share": q["a_site"]["number_share"],
    "q5_b_site_number_share": q["b_site"]["number_share"],
    "q6_b_site_exchange_ratio": q["b_site"]["exchange_ratio"],
    "q7_b_site_rho": q["b_site"]["b_to_a_coord_ratio"],
    "q8_b_site_apples_share": q["b_site"]["quantity_share"],
    "q11a_qwen_a_site_one_currency": q["a_site"]["headline"].startswith("one currency"),
    "q11b_qwen_b_site_one_currency": q["b_site"]["headline"].startswith("one currency"),
    "q11c_qwen_b_site_v5": v5key,
}
if "llama" in V:
    lf = json.load(open(res / "llama/fit.json"))
    lmp = json.load(open(res / "llama/map.json"))
    outcome["q9_llama_usable"] = bool(lf["usable"])
    nL = 32                                        # Llama 3.1 8B has 32 blocks
    arr = json.load(open(res / "llama/das.json"))["choice"]["arrival_layer"]
    outcome["q10_llama_arrival_fraction"] = arr / nL
    if lf["usable"]:
        outcome["q11d_llama_same_b_headline"] = V["llama"]["b_site"]["headline"].split(";")[0] == q["b_site"]["headline"].split(";")[0]
preds = {"Claude": yaml.safe_load(open(HERE / "predictions/claude_predictions.yaml")),
         "Jasper": yaml.safe_load(open(HERE / "predictions/predictions.yaml"))}
rows, brier = [], {k: [] for k in preds}
for qn, o in outcome.items():
    cells = []
    for who, P in preds.items():
        if qn not in P:
            cells.append("–")
            continue
        p = P[qn]
        if isinstance(p, dict) and "best" not in p:
            s = sum((v - (1.0 if k == o else 0.0)) ** 2 for k, v in p.items())
            brier[who].append(s)
            cells.append(f"{p.get(o, 0):.0%} on it; Brier {s:.2f}")
        elif isinstance(o, (bool, np.bool_)):
            s = (p - float(o)) ** 2
            brier[who].append(s)
            cells.append(f"{p:.0%}; Brier {s:.2f}")
        elif isinstance(p, dict):
            lo, hi = p["range80"]
            hit = lo <= o <= hi
            cells.append(f"{p['best']} [{lo}, {hi}] {'hit' if hit else 'miss'}, off by {abs(p['best'] - o):.2g}")
        else:
            cells.append(f"{p} {'right' if p == o else 'wrong'}")
    rows.append((qn, o, cells))
lines = ["# Pilot 7: predictions against results", "",
         "| question | result | Claude | Jasper |", "|---|---|---|---|"]
for qn, o, c in rows:
    oo = f"{o:.3g}" if isinstance(o, float) else str(o)
    lines.append(f"| {qn} | {oo} | {c[0]} | {c[1]} |")
lines += ["", "Mean Brier over the probability questions each answered (lower is better): " +
          ", ".join(f"{k} {sum(v) / len(v):.3f} ({len(v)} questions)" for k, v in brier.items() if v)]
common = [qn for qn, _, c in rows if c[1] != "–" and ("Brier" in c[0] and "Brier" in c[1])]
if common:
    cb = [float(c[0].split("Brier ")[1]) for qn, _, c in rows if qn in common]
    jb = [float(c[1].split("Brier ")[1]) for qn, _, c in rows if qn in common]
    lines.append(f"On the probability questions both answered ({len(common)}): Claude {np.mean(cb):.3f}, Jasper {np.mean(jb):.3f}")
for who in preds:
    j = list(preds).index(who)
    hits = sum(1 for _, _, c in rows if " hit" in c[j])
    nr = sum(1 for _, _, c in rows if "[" in c[j])
    lines.append(f"- {who}: {hits} of {nr} 80% ranges contain the result")
(res / "predictions_vs_results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
