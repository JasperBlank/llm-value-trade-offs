"""Stage `analyze`: the pre-registered verdicts (DESIGN.md, section 3) and a summary of the rest.
Writes results/<run>/report.md and verdicts.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import log, save_json


def load(res, name):
    p = Path(res) / name
    return json.load(open(p)) if p.exists() else None


def verdict_at(lay, C, n_shuf):
    a, b, r = lay["a"], lay["b"], lay["random"]
    causal = a["ci95"][0] > 0 and b["ci95"][1] < 0
    lim = C["control_max_ratio"] * a["slope"]
    random_ok = a["slope"] > 0 and abs(r["slope"]) < lim
    shuf_ok = a["slope"] > 0 and all(abs(lay[f"shuffled{j + 1}"]["slope"]) < lim for j in range(n_shuf))
    size_ok = C["min_slope"] <= a["slope"] <= C["max_slope"] and C["min_slope"] <= -b["slope"] <= C["max_slope"]
    if causal and random_ok and shuf_ok and size_ok:
        v = "used as value, at full size"
    elif causal and random_ok and shuf_ok:
        v = "used, but not at full size"
    elif causal:
        v = "an effect, but not specific to value (controls failed)"
    else:
        v = "not shown"
    return v, {"causal": causal, "random_ok": random_ok, "shuffled_ok": shuf_ok, "size_ok": size_ok}


def run(ctx):
    cfg = ctx.cfg
    res = cfg["res"]
    st, di = load(res, "steering.json"), load(res, "directions.json")
    C = cfg["steering"]["criteria"]
    n_shuf = int(cfg["steering"]["shuffles"])
    P = str(cfg["primary_layer"])
    lines = ["# Pilot 5 results", ""]
    V = {}
    if st:
        V["per_layer"] = {}
        for L, lay in st["per_layer"].items():
            v, parts = verdict_at(lay, C, n_shuf)
            V["per_layer"][L] = {"verdict": v, **parts}
        V["main"] = V["per_layer"][P]["verdict"]
        V["full_size_at_any_layer"] = any(x["verdict"] == "used as value, at full size" for x in V["per_layer"].values())
        cur = st["per_layer"][P]["currency"]
        r = float(C["currency_ratio"])
        V["currency"] = {k: (None if not np.isfinite(x.get("ratio", np.nan)) or not V["per_layer"][P]["causal"]
                             else bool(1 / r <= x["ratio"] <= r)) for k, x in cur.items()}
        lines += ["## The pre-registered test (steering at the last position)", "",
                  f"{st['n_prompts']} fresh prompts. Slope = change in log-odds of A per value unit added "
                  "(predicted: +1 for A's direction, -1 for B's, about 0 for the controls).", "",
                  "| layer | A's direction | B's direction | random | " +
                  " | ".join(f"shuffled {j + 1}" for j in range(n_shuf)) + " | verdict |",
                  "|---|---|---|---|" + "---|" * n_shuf + "---|"]
        for L, lay in st["per_layer"].items():
            f = lambda x: f"{x['slope']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]"
            lines.append(f"| {L}{' (primary)' if L == P else ''} | {f(lay['a'])} | {f(lay['b'])} | {f(lay['random'])} | "
                         + " | ".join(f(lay[f'shuffled{j + 1}']) for j in range(n_shuf)) + f" | {V['per_layer'][L]['verdict']} |")
        lines += ["", f"**Main verdict (layer {P}): {V['main']}.** Full size at any layer: "
                  f"{'yes' if V['full_size_at_any_layer'] else 'no'}.", "", "One currency (A's slope by kind of good, layer "
                  f"{P}; within a factor {C['currency_ratio']}):", ""]
        for k, x in cur.items():
            if "slopes" in x:
                lines.append(f"- {k.replace('_', ' ')}: slopes {x['slopes'][0]:+.2f} and {x['slopes'][1]:+.2f} "
                             f"(n = {x['n'][0]}, {x['n'][1]}), ratio {x['ratio']:.2f} "
                             f"[{x['ratio_ci95'][0]:.2f}, {x['ratio_ci95'][1]:.2f}]: "
                             f"{ {True: 'one currency', False: 'not one currency', None: 'not evaluated'}[V['currency'][k]]}")
    if di:
        lines += ["", "## The directions", "", "| layer | option-level r, A | option-level r, B | cos(w_A, w_B) | "
                  "steering length / residual length |", "|---|---|---|---|---|"]
        for L, x in di["per_layer"].items():
            lines.append(f"| {L} | {x['option_r_a']:.2f} | {x['option_r_b']:.2f} | {x['cos_wa_wb']:+.2f} | "
                         f"{x['steer_len_a']:.3g} / {x['mean_residual_len']:.3g} |")
        V["cos_wa_wb_primary"] = di["per_layer"][P]["cos_wa_wb"]
    ci = load(res, "circuit.json")
    if ci:
        w = ci["writers"][f"v_a@{P}"]
        wb = ci["writers"][f"v_b@{P}"]
        lines += ["", "## Exploratory: the circuit", "",
                  f"Writers of A's value at layer {P}: heads {w['heads_share']:.0%}, MLPs {w['mlps_share']:.0%}, "
                  f"embedding {w['embedding_share']:.0%}; half of it carried by {w['pieces_for_half']} of "
                  f"{w['n_pieces']} pieces (80% by {w['pieces_for_80pct']}). Overall number-to-good ratio "
                  f"{w['total_ratio_q_to_u'] if w['total_ratio_q_to_u'] is None else round(w['total_ratio_q_to_u'], 2)} (b = {ci['b']:.2f}).",
                  f"Writers of B's value: heads {wb['heads_share']:.0%}, MLPs {wb['mlps_share']:.0%}; half by {wb['pieces_for_half']} pieces.",
                  "", "Top writers of A's value:", "", "| piece | share | number part | good part | ratio |", "|---|---|---|---|---|"]
        for t in w["top"][:10]:
            lines.append(f"| {t['piece']} | {t['share_of_total']:.1%} | {t['beta_q']:+.3f} | {t['beta_u']:+.3f} | "
                         f"{'' if t['ratio_q_to_u'] is None else round(t['ratio_q_to_u'], 2)} |")
        if "all_heads_share_by_region" in w:
            top_regions = sorted(w["all_heads_share_by_region"].items(), key=lambda kv: -kv[1])[:4]
            lines += ["", "Where the heads read A's value from (share of the total): " +
                      ", ".join(f"{k} {v:.0%}" for k, v in top_regions)]
        r = ci["readers"]
        lines += ["", f"Readers (direct share of the log-odds): heads {r['heads_share']:.0%}, MLPs {r['mlps_share']:.0%}. "
                  f"Letter heads among the top {len(r['top_heads'])}: {', '.join(r['letter_heads']) or 'none'}.", "",
                  "| head | share | attention to A label | to B label | corr(label attention, log-odds) |", "|---|---|---|---|---|"]
        for h in r["top_heads"]:
            lines.append(f"| {h['piece']} | {h['share_of_logodds']:.1%} | {h['attention_a_label']:.2f} | "
                         f"{h['attention_b_label']:.2f} | {h['corr_label_attention_diff_with_logodds']:+.2f} |")
        V["exploratory"] = {"mlp_share_a": w["mlps_share"], "pieces_for_half_a": w["pieces_for_half"],
                            "letter_heads": r["letter_heads"]}
    pa = load(res, "patching.json")
    if pa:
        lines += ["", f"Patching ({pa['n_pairs']} pairs; transfer of the swapped good, 0 = none, 1 = all):", ""]
        for k, v in pa["conditions"].items():
            lines.append(f"- {k}: {v['transfer']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
        rl = [(int(k.split("@")[1]), v["transfer"]) for k, v in pa["conditions"].items() if k.startswith("resid@")]
        first = next((L for L, t in sorted(rl) if t >= 0.5), None)
        V.setdefault("exploratory", {})["first_layer_half_transfer"] = first
        V["exploratory"]["top20_transfer"] = pa["conditions"].get("top-20", {}).get("transfer")
    rd = load(res, "readers.json")
    if rd:
        lines += ["", f"Readers under the push (A's direction at layer {rd['layer']}, -2 to +2; log-odds shift "
                  f"{rd['logodds_shift']['-2.0']:+.2f} to {rd['logodds_shift']['2.0']:+.2f}):", ""]
        for k, v in rd["heads"].items():
            lines.append(f"- {k}: attention to A label {v['attention_a_label']['-2.0']:.2f} -> {v['attention_a_label']['2.0']:.2f}, "
                         f"B label {v['attention_b_label']['-2.0']:.2f} -> {v['attention_b_label']['2.0']:.2f}"
                         + ("" if v["after_push"] else " (before the push)"))
    ob = load(res, "objective.json")
    if ob:
        lines += ["", "## Exploratory: the objective, read at the last position", ""]
        for ver, x in ob.items():
            y = x["per_layer"][P]
            lines.append(f"- {ver}: the choice moves {x['choice_shift_toward_target']:+.1f} log-odds towards the target; "
                         f"at layer {P} the target's value readout moves {y['target_value_shift']:+.2f}, the partner's "
                         f"{y['partner_value_shift']:+.2f}, their difference {y['difference_shift_toward_target']:+.2f} "
                         f"({(y['fraction_of_choice_shift'] or 0):.0%} of the choice's move).")
    fo = load(res, "formats.json")
    if fo:
        Pp = str(fo["primary_layer"])
        lines += ["", "## Extension: other answer formats (the A/B arrows, applied elsewhere)", "",
                  "| format | behaviour | A's arrow readable (option-level r) | push A (slope, 95%) | relative to the format's behaviour | push B | random |",
                  "|---|---|---|---|---|---|---|"]
        for fm, x in fo["formats"].items():
            bh = x["behaviour"]
            beh = (f"corr with value {bh['corr_with_value_real']:.2f}" if fm == "rating"
                   else f"corr with value difference {bh['corr_with_value_difference']:.2f}")
            c = x["causal"]
            f = lambda k: (f"{c[k]['slope']:+.3f} [{c[k]['ci95'][0]:+.3f}, {c[k]['ci95'][1]:+.3f}]" if k in c else "")
            rel = c[f"a@{Pp}"].get("relative_to_behaviour")
            lines.append(f"| {fm} | {beh} | {x['readable'][Pp]['a_option_r']:.2f} | {f(f'a@{Pp}')} | "
                         f"{'' if rel is None else round(rel, 2)} | {f(f'b@{Pp}')} | {f(f'random@{Pp}')} |")
        w = fo.get("writers_numbered", {})
        lines += ["", f"Writers of A's value in the numbered format: {w.get('overlap_top20')} of the A/B top 20 also in its top 20; "
                  f"correlation of all pieces' shares with the A/B format: {w.get('corr_all_shares_with_ab')}; "
                  f"MLP share {w.get('mlps_share', float('nan')):.0%}."]
        rn = fo["formats"]["numbered"]["causal"][f"a@{Pp}"].get("relative_to_ab_slope")
        V["formats"] = {fm: {"a_option_r": x["readable"][Pp]["a_option_r"],
                             "push_a_slope": x["causal"][f"a@{Pp}"]["slope"],
                             "push_a_ci": x["causal"][f"a@{Pp}"]["ci95"]} for fm, x in fo["formats"].items()}
        V["formats"]["numbered_relative_to_ab"] = rn
        V["formats"]["writers_overlap_top20"] = w.get("overlap_top20")
    (Path(res) / "report.md").write_text("\n".join(lines) + "\n")
    save_json(V, Path(res) / "verdicts.json")
    log(f"analyze done: main verdict: {V.get('main')}")
