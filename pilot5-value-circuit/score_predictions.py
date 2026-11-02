"""Score Claude's and Jasper's pilot-5 predictions against the results (main set; the formats
extension is scored once it has run). Probabilities: Brier score (lower is better). Numbers: does
the 80% range contain the result, and the distance of the best guess."""
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
res = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/real")
st, di, ci = (json.load(open(res / f)) for f in ("steering.json", "directions.json", "circuit.json"))
ve, pa, ob = (json.load(open(res / f)) for f in ("verdicts.json", "patching.json", "objective.json"))
L = st["per_layer"]
slopes_a = {k: v["a"]["slope"] for k, v in L.items()}
w = ci["writers"]["v_a@28"]
b = ci["b"]
top10_mix = any(t["ratio_q_to_u"] is not None and b / 2 <= t["ratio_q_to_u"] <= 2 * b for t in w["top"][:10])
outcome = {
    "q1_a_slope_28": L["28"]["a"]["slope"], "q2_b_slope_28": L["28"]["b"]["slope"],
    "q3_controls_specific_28": ve["per_layer"]["28"]["random_ok"] and ve["per_layer"]["28"]["shuffled_ok"],
    "q4a_layer_largest_a_slope": int(max(slopes_a, key=slopes_a.get)), "q4b_full_size_any_layer": ve["full_size_at_any_layer"],
    "q5_cos_wa_wb_28": di["per_layer"]["28"]["cos_wa_wb"],
    "q6a_currency_gain_vs_harm": ve["currency"]["gain_worded_vs_harm_worded"],
    "q6b_currency_money_vs_other": ve["currency"]["money_vs_nonmoney"],
    "q7_mlp_share_a_28": w["mlps_share"], "q8_pieces_for_half": w["pieces_for_half"],
    "q9_first_layer_half_transfer": ve["exploratory"]["first_layer_half_transfer"],
    "q10_top20_transfer": pa["conditions"]["top-20"]["transfer"],
    "q11_letter_head": len(ci["readers"]["letter_heads"]) > 0, "q12_number_and_good_in_one_piece": top10_mix,
    "q13_goal1_fraction_28": ob["goal1"]["per_layer"]["28"]["fraction_of_choice_shift"],
    "q14_verdict_28": {"used as value, at full size": "full_size", "used, but not at full size": "used_not_full",
                       "an effect, but not specific to value (controls failed)": "not_specific",
                       "not shown": "not_shown"}[ve["main"]],
}
preds = {"Claude": yaml.safe_load(open(HERE / "predictions/claude_predictions.yaml")),
         "Jasper": yaml.safe_load(open(HERE / "predictions/predictions.yaml"))}
rows, brier = [], {k: [] for k in preds}
for q, o in outcome.items():
    cells = []
    for who, P in preds.items():
        p = P[q]
        if q == "q14_verdict_28":
            s = sum((v - (1.0 if k == o else 0.0)) ** 2 for k, v in p.items())
            brier[who].append(s)
            cells.append(f"{p[o]:.0%} on it; Brier {s:.2f}")
        elif isinstance(o, bool):
            pp = p if not isinstance(p, dict) else p
            s = (pp - float(o)) ** 2
            brier[who].append(s)
            cells.append(f"{pp:.0%}; Brier {s:.2f}")
        elif isinstance(p, dict):
            lo, hi = p["range80"]
            hit = lo <= o <= hi
            extra = ""
            if "p_above_half" in p:
                s = (p["p_above_half"] - float(o > 0.5)) ** 2
                brier[who].append(s)
                extra = f"; above half {p['p_above_half']:.0%}, Brier {s:.2f}"
            cells.append(f"{p['best']} [{lo}, {hi}] {'hit' if hit else 'miss'}, off by {abs(p['best'] - o):.2g}{extra}")
        else:
            cells.append(f"{p} {'right' if p == o else 'wrong'}")
    rows.append((q, o, cells))
lines = ["# Pilot 5: predictions against results (main set)", "",
         "| question | result | Claude | Jasper |", "|---|---|---|---|"]
for q, o, c in rows:
    oo = f"{o:.3g}" if isinstance(o, float) else str(o)
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

# ---------------------------------------------------------------- the formats extension
fs = HERE / "results/formats_run/formats_summary.json"
if fs.exists():
    F = json.load(open(fs))
    fo = {
        "f1_rating_behaviour_corr": F["rating"]["behaviour"]["corr_with_value_real"],
        "f2_rating_option_r": F["rating"]["readable"]["28"]["a_option_r"],
        "f3_numbered_relative_to_ab": F["numbered"]["causal"]["a@28"]["relative_to_ab_slope"],
        "f4_should_push_raises_yes": F["should"]["causal"]["a@28"]["ci95"][0] > 0,
        "f5_rating_push_raises_rating": F["rating"]["causal"]["a@28"]["ci95"][0] > 0,
        "f6_writers_overlap_top20": F["writers_numbered"]["overlap_top20"],
    }
    fp = {"Claude": yaml.safe_load(open(HERE / "predictions/claude_formats.yaml")),
          "Jasper": yaml.safe_load(open(HERE / "predictions/predictions.yaml"))}
    out = ["", "# The formats extension", "", "| question | result | Claude | Jasper |", "|---|---|---|---|"]
    fb = {k: [] for k in fp}
    for q, o in fo.items():
        cells = []
        for who, P in fp.items():
            p = P[q]
            if isinstance(o, bool):
                s = (p - float(o)) ** 2
                fb[who].append(s)
                cells.append(f"{p:.0%}; Brier {s:.2f}")
            else:
                lo, hi = p["range80"]
                cells.append(f"{p['best']} [{lo}, {hi}] {'hit' if lo <= o <= hi else 'miss'}, off by {abs(p['best'] - o):.2g}")
        out.append(f"| {q} | {o if isinstance(o, bool) else round(o, 3)} | {cells[0]} | {cells[1]} |")
    out.append("")
    out.append("Mean Brier (F4, F5): " + ", ".join(f"{k} {sum(v) / len(v):.3f}" for k, v in fb.items()))
    with open(res / "predictions_vs_results.md", "a") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
