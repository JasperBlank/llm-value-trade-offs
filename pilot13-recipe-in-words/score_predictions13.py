"""Score Claude's and Jasper's pilot-13 predictions: ranges (hit if the result lies inside the 80% range) and
probabilities (Brier scores; lower is better)."""
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real")
V = json.load(open(res / "verdicts13.json"))
outcome = {"p1_features_r": V["P1"]["r"], "p2_moral_rating_prob": V["P2"]["pass"], "p3_beats_random_prob": V["P3"]["pass"],
           "p4_new_sentences_r": V["P4"]["r"]}
preds = {"Claude": yaml.safe_load(open(HERE / "predictions/claude_predictions.yaml")),
         "Jasper": yaml.safe_load(open(HERE / "predictions/predictions.yaml"))}
L = ["# Pilot 13: predictions against results", "", "| question | result | Claude | Jasper |", "|---|---|---|---|"]
for q, o in outcome.items():
    cells = []
    for P in preds.values():
        p = P.get(q)
        if p is None:
            cells.append("–")
        elif isinstance(p, dict):
            lo, hi = p["range80"]
            cells.append(f"{p['best']} [{lo}, {hi}]: {'hit' if lo <= o <= hi else 'miss'}, off by {abs(p['best'] - o):.2g}")
        else:
            cells.append(f"{p:.0%}; Brier {(p - float(o)) ** 2:.3f}")
    shown = f"{o:.2f}" if isinstance(o, float) else str(o)
    L.append(f"| {q} | {shown} | {cells[0]} | {cells[1]} |")
out = "\n".join(L) + "\n"
(res / "predictions_vs_results.md").write_text(out)
print(out)
