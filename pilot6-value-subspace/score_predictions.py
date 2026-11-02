"""Score Claude's and Jasper's pilot-6 predictions against the results. Probabilities: Brier score
(lower is better; the three-way P1 verdict summed over its three options). Numbers: does the 80%
range contain the result, and the distance of the best guess."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real")
C, D, Dd, St, V = (json.load(open(res / f)) for f in ("control.json", "das.json", "directions6.json", "steer6.json", "verdicts6.json"))
L = "28"
runs = D["layers"][L]["runs"]
k_ok = [k for k in (1, 2, 4, 8, 16) if runs[f"whole_k{k}"]["of_whole_swap"]["value"] >= 0.9]
p1 = V["P1"]["verdict"]
outcome = {
    "q1_quantity_half_layer": C["tasks"]["quantity"]["first_layer_half"],
    "q2_weight_half_layer": C["tasks"]["weight"]["first_layer_half"],
    "q3_quantity_top20_transfer": C["tasks"]["quantity"]["conditions"]["top20"]["value"],
    "q4_weight_top20_transfer": C["tasks"]["weight"]["conditions"]["top20"]["value"],
    "q5_das_k1_share_28": runs["whole_k1"]["of_whole_swap"]["value"],
    "q6_smallest_k_for_0.9": k_ok[0] if k_ok else "none",
    "q7_probe_swap_share_28": D["layers"][L]["probe_swap"]["of_whole_swap"]["value"],
    "q8_abs_cos_learned_probe_28": abs(Dd[f"{L}|whole_k1"]["cos_with_probe"]),
    "q9_corr_learned_va_heldout_28": Dd[f"{L}|whole_k1"]["corr_with_va_heldout"],
    "q10_push_ratio_learned_over_probe_28": St["directions"]["learned"]["slope_logit_diff"] / St["directions"]["probe"]["slope_logit_diff"],
    "q11a_p1_verdict": "generic" if p1.startswith("generic") else ("value_specific" if p1.startswith("value") else "mixed"),
    "q11b_p2_pass": V["P2"]["pass"], "q11c_p3_pass": V["P3"]["pass"], "q11d_p4_pass": V["P4"]["pass"],
}
preds = {"Claude": yaml.safe_load(open(HERE / "predictions/claude_predictions.yaml")),
         "Jasper": yaml.safe_load(open(HERE / "predictions/predictions.yaml"))}
rows, brier = [], {k: [] for k in preds}
for q, o in outcome.items():
    cells = []
    for who, P in preds.items():
        p = P[q]
        if q == "q11a_p1_verdict":
            s = sum((v - (1.0 if k == o else 0.0)) ** 2 for k, v in p.items())
            brier[who].append(s)
            cells.append(f"{p[o]:.0%} on it; Brier {s:.2f}")
        elif isinstance(o, bool):
            s = (p - float(o)) ** 2
            brier[who].append(s)
            cells.append(f"{p:.0%}; Brier {s:.2f}")
        elif isinstance(p, dict):
            lo, hi = p["range80"]
            hit = lo <= o <= hi
            cells.append(f"{p['best']} [{lo}, {hi}] {'hit' if hit else 'miss'}, off by {abs(p['best'] - o):.2g}")
        else:
            cells.append(f"{p} {'right' if p == o else 'wrong'}")
    rows.append((q, o, cells))
lines = ["# Pilot 6: predictions against results", "",
         "| question | result | Claude | Jasper |", "|---|---|---|---|"]
for q, o, c in rows:
    oo = f"{o:.3g}" if isinstance(o, (float, np.floating)) else str(o)
    lines.append(f"| {q} | {oo} | {c[0]} | {c[1]} |")
lines += ["", "Mean Brier over the probability questions (lower is better): " +
          ", ".join(f"{k} {sum(v) / len(v):.3f} ({len(v)} questions)" for k, v in brier.items())]
for who in preds:
    j = list(preds).index(who)
    hits = sum(1 for _, _, c in rows if " hit" in c[j])
    nr = sum(1 for _, _, c in rows if "[" in c[j])
    lines.append(f"- {who}: {hits} of {nr} 80% ranges contain the result")
(res / "predictions_vs_results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
