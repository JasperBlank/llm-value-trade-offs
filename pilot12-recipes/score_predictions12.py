"""Score Claude's and Jasper's pilot-12 predictions (Brier scores; lower is better)."""
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real")
V = json.load(open(res / "verdicts12.json"))
outcome = {"q1_disagreement_readable": V["Q1"]["pass"], "q2_disagreement_push_excess": V["Q2"]["pass"]}
preds = {"Claude": yaml.safe_load(open(HERE / "predictions/claude_predictions.yaml")),
         "Jasper": yaml.safe_load(open(HERE / "predictions/predictions.yaml"))}
L = ["# Pilot 12: predictions against results", "", "| question | result | Claude | Jasper |", "|---|---|---|---|"]
for q, o in outcome.items():
    cells = [f"{P[q]:.0%}; Brier {(P[q] - float(o)) ** 2:.3f}" if q in P else "–" for P in preds.values()]
    L.append(f"| {q} | {o} | {cells[0]} | {cells[1]} |")
out = "\n".join(L) + "\n"
(res / "predictions_vs_results.md").write_text(out)
print(out)
