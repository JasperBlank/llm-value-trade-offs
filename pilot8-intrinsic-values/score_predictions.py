"""Score Claude's and Jasper's pilot-8 predictions against the results. The top value: the rank the
model gives to the value each of us picked. Probabilities: Brier score (lower is better)."""
import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real")
V = json.load(open(res / "verdicts8.json"))
report = open(res / "report.md").read()
ranking = re.findall(r"^\| (\d+) \| ([A-Za-z-]+)(?: \(held out\))? \|", report, flags=re.M)
rank = {v: int(r) for r, v in ranking}
top = min(rank, key=rank.get)

outcome = {
    "q1_top_value": top,
    "q2_menu_consistent": V["Q3"]["one_scale"] and V["Q3"]["transitive"] and V["Q3"]["menu_invariant"],
    "q3_same_thing_lights_up": V["Q2a"]["pass"] and V["Q2b"]["pass"],
    "q4_ratings_match": V["Q4a"]["pass"],
    "q5_same_value_in_rating": V["Q4b"]["pass"],
}
preds = {"Claude": yaml.safe_load(open(HERE / "predictions/claude_predictions.yaml")),
         "Jasper": yaml.safe_load(open(HERE / "predictions/predictions.yaml"))}
brier = {k: [] for k in preds}
lines = ["# Pilot 8: predictions against results", "",
         "| question | result | Claude | Jasper |", "|---|---|---|---|"]
for qn, o in outcome.items():
    cells = []
    for who, P in preds.items():
        p = P[qn]
        if qn == "q1_top_value":
            cells.append(f"{p}: ranked {rank[p]} of {len(rank)}")
        else:
            s = (p - float(o)) ** 2
            brier[who].append(s)
            cells.append(f"{p:.0%}; Brier {s:.3f}")
    lines.append(f"| {qn} | {o} | {cells[0]} | {cells[1]} |")
lines += ["", "Mean Brier over the four probability questions (lower is better; always saying 50% scores 0.250): " +
          ", ".join(f"{k} {sum(v) / len(v):.3f}" for k, v in brier.items()), "",
          "Claude's sealed file also gave probabilities for the top value: Longevity 0.35, Protection 0.20, "
          "Non-suffering 0.15, Justice 0.07, Fairness 0.07, any other value 0.16. "
          f"The top value, {top}, falls under 'any other value'."]
out = "\n".join(lines) + "\n"
(res / "predictions_vs_results.md").write_text(out)
print(out)
