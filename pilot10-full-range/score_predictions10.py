"""Score Claude's and Jasper's pilot-10 predictions. q2 is pilot 9's follow-up question (Claude's sealed answer is
pilot9-vibe-check/predictions/claude_followup.yaml)."""
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real")
V = json.load(open(res / "verdicts10.json"))
outcome = {"q1_full_range_r": V["F1"]["r"], "q2_logscale_r": V["F2"]["r"], "q3_size_at_least_one_point": V["F3"]["size_sensitive"]}
cl = yaml.safe_load(open(HERE / "predictions/claude_predictions.yaml"))
cl["q2_logscale_r"] = yaml.safe_load(open(HERE.parent / "pilot9-vibe-check/predictions/claude_followup.yaml"))["followup_logscale_r"]
preds = {"Claude": cl, "Jasper": yaml.safe_load(open(HERE / "predictions/predictions.yaml"))}
L = ["# Pilot 10: predictions against results", "", "| question | result | Claude | Jasper |", "|---|---|---|---|"]
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
