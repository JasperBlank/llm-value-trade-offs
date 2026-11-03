"""Score Claude's and Jasper's pilot-11 predictions."""
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real")
V = json.load(open(res / "verdicts11.json"))
outcome = {"q1_readable_r": V["Q1"]["r"], "q2_overlap_prob": V["Q2"]["pass"], "q3_used_for_both_prob": V["Q3"]["pass"]}
preds = {"Claude": yaml.safe_load(open(HERE / "predictions/claude_predictions.yaml")),
         "Jasper": yaml.safe_load(open(HERE / "predictions/predictions.yaml"))}
L = ["# Pilot 11: predictions against results", "", "| question | result | Claude | Jasper |", "|---|---|---|---|"]
for q, o in outcome.items():
    cells = []
    for who, P in preds.items():
        p = P.get(q)
        if p is None:
            cells.append("–")
        elif isinstance(p, dict):
            lo, hi = p["range80"]
            cells.append(f"{p['best']} [{lo}, {hi}]: {'hit' if lo <= o <= hi else 'miss'}, off by {abs(p['best'] - o):.2g}")
        else:
            cells.append(f"{p:.0%}; Brier {(p - float(o)) ** 2:.3f}")
    L.append(f"| {q} | {o:.2f} | {cells[0]} | {cells[1]} |" if isinstance(o, float) else f"| {q} | {o} | {cells[0]} | {cells[1]} |")
out = "\n".join(L) + "\n"
(res / "predictions_vs_results.md").write_text(out)
print(out)
