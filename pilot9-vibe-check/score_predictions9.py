"""Score Claude's and Jasper's pilot-9 predictions. Numbers: does the 80% range contain the result, and how far
off the best guess is. The probability: Brier score (lower is better)."""
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real")
V = json.load(open(res / "verdicts9.json"))
outcome = {"t1_rating_vs_menu_r": V["T1"]["r"], "t2_same_sentence_count": V["T2"]["same"],
           "t3_size_at_least_half_point": V["T3"]["size_sensitive"]}
preds = {"Claude": yaml.safe_load(open(HERE / "predictions/claude_predictions.yaml")),
         "Jasper": yaml.safe_load(open(HERE / "predictions/predictions.yaml"))}
L = ["# Pilot 9: predictions against results", "", "| question | result | Claude | Jasper |", "|---|---|---|---|"]
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
    oo = f"{o:.2f}" if isinstance(o, float) else str(o)
    L.append(f"| {q} | {oo} | {cells[0]} | {cells[1]} |")
out = "\n".join(L) + "\n"
(res / "predictions_vs_results.md").write_text(out)
print(out)
