"""Pilot 3 analysis: the gate, the extras and the goal module, per model (DESIGN.md).

  python analyze.py            (every model with results)

Every number is computed on CELLS: one comparison averaged over both option orders and all
wordings that pass the compliance rule. Writes analysis_<model>.json, report_<model>.md and
summary.md.
"""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np

from common import GROUPS, goods_in, load_config, load_jsonl, log, out_path, parse_args, save_json
from fit import (UtilityFit, cell_p, cells, cv_correlation, heldout_rate_misses, miss_factor, pair_rate,
                 quantity_slope, transitivity_factor)

MODELS = ["14b", "8b", "32b", "quick"]


def most(n, total, share) -> bool:
    """At least `share` (e.g. "2/3") of `total`, compared exactly: 2 of 3 counts."""
    return total > 0 and Fraction(int(n)) >= Fraction(str(share)) * total


def geo_mean(xs):
    xs = [x for x in xs if np.isfinite(x) and x > 0]
    return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")


def compliance(cfg, rows):
    """Mean probability on the letters per wording; wordings under the bar are dropped."""
    by = {w: [r["mass"] for r in rows if r["phrasing"] == w] for w in range(cfg["wordings"])}
    mass = {w: float(np.mean(v)) if v else float("nan") for w, v in by.items()}
    kept = [w for w, v in mass.items() if np.isfinite(v) and v >= cfg["compliance"]["min_wording"]]
    return mass, kept


def reliability(rows, kinds, ben, kept):
    """Split-half reliability of cell probabilities across wordings (Spearman-Brown corrected)."""
    C = cells([r for r in rows if r["beneficiary"] == ben], kinds=kinds, wordings=kept, by_wording=True)
    by = {}
    for c in C:
        by.setdefault((c["x"], c["qx"], c["y"], c["qy"]), {})[c["wording"]] = c["p"]
    half1 = [w for w in kept if w % 2 == 0]
    half2 = [w for w in kept if w % 2 == 1]
    h1, h2 = [], []
    for v in by.values():
        if all(w in v for w in kept):
            h1.append(np.mean([v[w] for w in half1]))
            h2.append(np.mean([v[w] for w in half2]))
    if len(h1) < 3 or not half1 or not half2:
        return float("nan")
    r = np.corrcoef(h1, h2)[0, 1]
    return float(2 * r / (1 + r))


def gate(cfg, C, ben):
    G = cfg["gate"]
    gains, harms = goods_in(cfg, "gain"), goods_in(cfg, "harm_prevented")
    reals = [g["id"] for g in gains + harms]
    neutrals = [g["id"] for g in goods_in(cfg, "neutral")]
    real_cells = [c for c in C if c["kind"] in ("within", "cross") and c["beneficiary"] == ben]

    slopes = {g: quantity_slope(real_cells, g) for g in reals}
    c1a = all(np.isfinite(b) and np.isfinite(se) and b - G["slope_z"] * se > 0 for b, se, _ in slopes.values())

    rvn = [c for c in C if c["kind"] == "rvn" and c["beneficiary"] == ben]
    p_real = float(np.mean([c["p"] if c["x"] in reals else 1 - c["p"] for c in rvn])) if rvn else float("nan")
    p_real_1v8 = float(np.mean([cell_p(rvn, r, 1, n, 8) for r in reals for n in neutrals]))
    c1b = bool(p_real >= G["real_vs_neutral_min"])

    pairs = [(a, b) for a, b in itertools.combinations([g["id"] for g in gains], 2)] + \
            [(a, b) for a, b in itertools.combinations([g["id"] for g in harms], 2)] + \
            [(g["id"], h["id"]) for g in gains for h in harms]
    misses = heldout_rate_misses(real_cells, pairs)
    m_vals = [v["miss"] for v in misses.values()]
    median_miss = float(np.median(m_vals))
    within2 = sum(v <= G["c2_share_factor"] for v in m_vals)
    c2 = bool(median_miss <= G["c2_median_max"] and most(within2, len(m_vals), G["c2_share"]))

    within = [c for c in real_cells if c["kind"] == "within"]
    tf = {"gain": transitivity_factor(within, [g["id"] for g in gains]),
          "harm_prevented": transitivity_factor(within, [g["id"] for g in harms])}
    c3 = all(v <= G["transitivity_max_factor"] for v in tf.values())
    return {"1a": bool(c1a), "1b": c1b, "2": c2, "3": bool(c3), "pass": bool(c1a and c1b and c2 and c3),
            "slopes": {g: {"slope": b, "se": se, "n": n} for g, (b, se, n) in slopes.items()},
            "p_real_over_neutral": p_real, "p_real_1_vs_8_neutral": p_real_1v8,
            "rate_misses": misses, "median_miss": median_miss, "pairs_within_x2": within2, "n_pairs": len(m_vals),
            "transitivity": tf, "heldout_r": cv_correlation(real_cells)}


def extras(cfg, C, rows, ben, kept, gate_res):
    E = cfg["extras"]
    gains, harms = goods_in(cfg, "gain"), goods_in(cfg, "harm_prevented")
    out = {}
    real_cells = [c for c in C if c["kind"] in ("within", "cross") and c["beneficiary"] == ben]
    # Description invariance: the gain wording of each prevented harm against its usual wording.
    fit = UtilityFit(real_cells + [c for c in C if c["kind"] == "mirror" and c["beneficiary"] == ben])
    desc = {h["id"]: miss_factor(fit.rate(h["id"] + "~gainword", h["id"]), 1.0) for h in harms}
    out["description"] = {"factors": desc,
                          "pass": most(sum(v <= E["invariance_max_factor"] for v in desc.values()), len(desc), E["pass_share"])}
    # Loss aversion: preventing a harm against giving the gain in the same domain.
    fit0 = UtilityFit(real_cells)
    by_dom = {h["domain"]: fit0.rate(h["id"], next(g["id"] for g in gains if g["domain"] == h["domain"])) for h in harms}
    out["loss_aversion"] = {"by_domain": by_dom, "geometric_mean": geo_mean(by_dom.values())}
    # Menu invariance at the balance point: P(a)/(P(a)+P(b)) with a third option, against the
    # two-option probability in the same cell.
    two = [c for c in C if c["kind"] == "within" and c["beneficiary"] == ben]
    groups = {}
    for r in rows:
        if r["kind"] != "menu" or r["beneficiary"] != ben or r["phrasing"] not in kept:
            continue
        a, b = r["focal"]
        qa, qb = r["focal_q"]
        ia = next(i for i, o in enumerate(r["options"]) if o["good"] == a and o["q"] == qa)
        ib = next(i for i, o in enumerate(r["options"]) if o["good"] == b and o["q"] == qb)
        s = r["probs"][ia] + r["probs"][ib]
        if s > 0:
            groups.setdefault((a, qa, b, qb, r["third_type"]), []).append(r["probs"][ia] / s)
    m_fac = {}
    for (a, qa, b, qb, tt), shares in groups.items():
        p2, p3 = cell_p(two, a, qa, b, qb), float(np.mean(shares))
        odds = (lambda p: np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        if np.isfinite(p2):
            m_fac[f"{a}:{qa} vs {b}:{qb}, {tt}"] = {"two_option": p2, "with_third": p3,
                                                     "factor": miss_factor(odds(p3), odds(p2))}
    out["menu"] = {"menus": m_fac,
                   "pass": most(sum(v["factor"] <= E["invariance_max_factor"] for v in m_fac.values()), len(m_fac),
                                E["pass_share"])}
    # Reported: the neutral-versus-neutral pull of numbers, and reliability across wordings.
    neu = [c for c in C if c["kind"] == "neutral" and c["beneficiary"] == ben]
    n_slopes = {n["id"]: quantity_slope(neu, n["id"])[0] for n in goods_in(cfg, "neutral")}
    med_real = float(np.median([v["slope"] for v in gate_res["slopes"].values()]))
    out["neutral_slopes"] = n_slopes
    out["neutral_slope_ratio"] = float(np.median([abs(v) for v in n_slopes.values()]) / med_real) if med_real > 0 else float("nan")
    out["reliability_across_wordings"] = reliability(rows, ("within", "cross"), ben, kept)
    return out


def closeness(cfg, C):
    E = cfg["extras"]
    mixed = [c for c in C if c["kind"] == "mixed"]
    reals = [g["id"] for g in goods_in(cfg, "gain") + goods_in(cfg, "harm_prevented")]
    mult = {g: pair_rate(mixed, f"{g}@close", f"{g}@strangers") for g in reals}
    finite = [v for v in mult.values() if np.isfinite(v) and v > 0]
    med = float(np.median(finite)) if finite else float("nan")
    uniform = bool(finite) and all(miss_factor(v, med) <= E["invariance_max_factor"] for v in mult.values())
    return {"multipliers": mult, "median": med, "uniform": bool(uniform)}


def goal_module(cfg, C, G, gate_res, ben):
    GM = cfg["goal_module"]
    gains, harms = goods_in(cfg, "gain"), goods_in(cfg, "harm_prevented")
    archive = next(n["id"] for n in goods_in(cfg, "neutral") if n["domain"] == "archive")
    ref = [c for c in C if c["kind"] in ("within", "cross") and c["beneficiary"] == ben]
    G = [c for c in G if c["beneficiary"] == ben]
    t_lo = min(cfg["goal"]["v1_target"])

    v1 = {}
    for t in gains + harms:
        for p in (gains if t["valence"] == "gain" else harms):
            if p["id"] == t["id"]:
                continue
            g_c = [c for c in G if c["kind"] == "goal1" and c["goal_good"] == t["id"] and {c["x"], c["y"]} == {t["id"], p["id"]}]
            shift = pair_rate(g_c, t["id"], p["id"]) / pair_rate(ref, t["id"], p["id"])
            s_goal = quantity_slope(g_c, p["id"])[0]
            s_ref = gate_res["slopes"][p["id"]]["slope"]
            ratio = s_goal / s_ref if s_ref > 0 else float("nan")
            p8, p32, p128 = (cell_p(g_c, t["id"], t_lo, p["id"], q) for q in (8, 32, 128))
            v1[f"{t['id']} vs {p['id']}"] = {"value_shift": shift, "p_target_1_vs_8": p8, "p_target_1_vs_32": p32,
                                             "p_target_1_vs_128": p128,
                                             "partner_slope_ratio": ratio,
                                             "rule_like": bool(p8 >= GM["rule_min_prob"] and ratio < GM["rule_max_slope_ratio"])}
    v2 = {}
    for r in gains + harms:
        g_c = [c for c in G if c["kind"] == "goal2" and {c["x"], c["y"]} == {archive, r["id"]}]
        p8, p32, p128 = (cell_p(g_c, archive, 1, r["id"], q) for q in (8, 32, 128))
        price = float("inf") if p128 > 0.5 else pair_rate(g_c, archive, r["id"])
        s_ref = gate_res["slopes"][r["id"]]["slope"]
        ratio = quantity_slope(g_c, r["id"])[0] / s_ref if s_ref > 0 else float("nan")
        v2[r["id"]] = {"welfare_price": price, "p_neutral_1_vs_8": p8, "p_neutral_1_vs_32": p32,
                       "p_neutral_1_vs_128": p128,
                       "partner_slope_ratio": ratio,
                       "rule_like": bool(p8 >= GM["rule_min_prob"] and ratio < GM["rule_max_slope_ratio"])}
    share = cfg["extras"]["pass_share"]
    return {"version1": {"pairs": v1, "rule_like": most(sum(v["rule_like"] for v in v1.values()), len(v1), share),
                         "median_value_shift": float(np.nanmedian([v["value_shift"] for v in v1.values()]))},
            "version2": {"goods": v2, "rule_like": most(sum(v["rule_like"] for v in v2.values()), len(v2), share),
                         "median_welfare_price": float(np.median([v["welfare_price"] for v in v2.values()]))}}


def f2(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if x == float("inf"):
        return "infinite"
    if abs(x) >= 1000:
        return "over 1000"
    return f"{x:.2f}" if abs(x) < 100 else f"{x:.0f}"


def fprice(x):
    return "above 128" if x == float("inf") else f2(x)


def report(m, res):
    gs = res["groups"]
    yn = (lambda v: "pass" if v else "fail")
    md = [f"# Pilot 3 results: {m}", "",
          f"Wordings kept (≥ 95% on the letters): {res['wordings_kept']} "
          f"(probability on the letters per wording: {', '.join(f'{v:.1%}' for v in res['letter_mass'].values())}).", "",
          "| criterion | strangers | family |", "|---|---|---|",
          "| 1a quantity sensitivity, all real goods | " + " | ".join(yn(gs[b]["gate"]["1a"]) for b in GROUPS) + " |",
          "| 1b real over neutral (mean probability) | " + " | ".join(f"{gs[b]['gate']['p_real_over_neutral']:.2f}" for b in GROUPS) + " |",
          "| 1b | " + " | ".join(yn(gs[b]["gate"]["1b"]) for b in GROUPS) + " |",
          "| 2 median held-out exchange-rate miss | " + " | ".join(f"×{f2(gs[b]['gate']['median_miss'])}" for b in GROUPS) + " |",
          "| 2 pairs within ×2 | " + " | ".join(f"{gs[b]['gate']['pairs_within_x2']} of {gs[b]['gate']['n_pairs']}" for b in GROUPS) + " |",
          "| 2 | " + " | ".join(yn(gs[b]["gate"]["2"]) for b in GROUPS) + " |",
          "| 3 transitivity factor (gains / harms prevented) | " + " | ".join(
              f"{f2(gs[b]['gate']['transitivity']['gain'])} / {f2(gs[b]['gate']['transitivity']['harm_prevented'])}" for b in GROUPS) + " |",
          "| 3 | " + " | ".join(yn(gs[b]["gate"]["3"]) for b in GROUPS) + " |",
          "| **Gate** | " + " | ".join(f"**{yn(gs[b]['gate']['pass'])}**" for b in GROUPS) + " |", "",
          "Quantity slopes per real good (strangers): " + ", ".join(
              f"{g} {f2(v['slope'])} ± {f2(v['se'])}" for g, v in gs["strangers"]["gate"]["slopes"].items()), "",
          "## Reported alongside", "",
          "| | strangers | family |", "|---|---|---|",
          "| held-out correlation (pilot 2's criterion 2) | " + " | ".join(f2(gs[b]["gate"]["heldout_r"]) for b in GROUPS) + " |",
          "| reliability across wordings | " + " | ".join(f2(gs[b]["extras"]["reliability_across_wordings"]) for b in GROUPS) + " |",
          "| real over neutral at 1 against 8 | " + " | ".join(f2(gs[b]["gate"]["p_real_1_vs_8_neutral"]) for b in GROUPS) + " |",
          "| neutral-vs-neutral slope ÷ real slope | " + " | ".join(f2(gs[b]["extras"]["neutral_slope_ratio"]) for b in GROUPS) + " |",
          "", "## Extras", "", "| extra | strangers | family |", "|---|---|---|"]
    ex = {b: gs[b]["extras"] for b in GROUPS}
    md.append("| menu invariance | " + " | ".join(
        f"{yn(ex[b]['menu']['pass'])} (median factor {f2(np.median([v['factor'] for v in ex[b]['menu']['menus'].values()] or [np.nan]))})"
        for b in GROUPS) + " |")
    md.append("| description invariance | " + " | ".join(
        f"{yn(ex[b]['description']['pass'])} (" + ", ".join(f2(v) for v in ex[b]["description"]["factors"].values()) + ")"
        for b in GROUPS) + " |")
    md.append("| loss aversion (geometric mean) | " + " | ".join(f2(ex[b]["loss_aversion"]["geometric_mean"]) for b in GROUPS) + " |")
    c = res["closeness"]
    md += ["", f"Closeness: family ÷ strangers multiplier, median {f2(c['median'])}; uniform: "
           f"{'yes' if c['uniform'] else 'no'} (" + ", ".join(f"{g} {f2(v)}" for g, v in c["multipliers"].items()) + ")."]
    if res["goal"]:
        gm = {b: gs[b]["goal"] for b in GROUPS}
        md += ["", "## Goal module", "", "| | strangers | family |", "|---|---|---|",
               "| objective on a real good: rule-like | " + " | ".join(
                   ("yes" if gm[b]["version1"]["rule_like"] else "no, extra value")
                   + f" (median value shift {f2(gm[b]['version1']['median_value_shift'])}×)" for b in GROUPS) + " |",
               "| objective on the archive good: rule-like | " + " | ".join(
                   ("yes" if gm[b]["version2"]["rule_like"] else "no")
                   + f" (median welfare price {fprice(gm[b]['version2']['median_welfare_price'])} people per file)" for b in GROUPS) + " |",
               "", "Welfare price per real good (strangers; people one file is worth): " + ", ".join(
                   f"{g} {fprice(v['welfare_price'])}" for g, v in gm["strangers"]["version2"]["goods"].items())]
    return "\n".join(md) + "\n"


def analyse_model(cfg, m):
    rows = load_jsonl(out_path(cfg, f"test_{m}.jsonl"))
    if not rows:
        return None
    goal_rows = load_jsonl(out_path(cfg, f"goal_{m}.jsonl"))
    log(f"Analysing {m} ({len(rows)} test prompts, {len(goal_rows)} goal prompts)")
    mass, kept = compliance(cfg, rows + goal_rows)
    if not kept:
        msg = f"# Pilot 3 results: {m}\n\nNo wording put at least 95% of the probability on the letters: {mass}. Nothing analysed.\n"
        out_path(cfg, f"report_{m}.md").write_text(msg, encoding="utf-8")
        log(f"{m}: no wording passed the compliance rule; skipped")
        return None
    C = cells(rows, wordings=kept)
    G = cells(goal_rows, wordings=kept)
    res = {"model": m, "letter_mass": mass, "wordings_kept": kept, "groups": {}, "goal": bool(goal_rows)}
    for ben in GROUPS:
        g = gate(cfg, C, ben)
        res["groups"][ben] = {"gate": g, "extras": extras(cfg, C, rows, ben, kept, g)}
        if goal_rows:
            res["groups"][ben]["goal"] = goal_module(cfg, C, G, g, ben)
    res["closeness"] = closeness(cfg, C)
    save_json(res, out_path(cfg, f"analysis_{m}.json"))
    out_path(cfg, f"report_{m}.md").write_text(report(m, res), encoding="utf-8")
    return res


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    summary = ["# Pilot 3: summary over models", "",
               "| model | gate (strangers) | gate (family) | median rate miss (strangers) | transitivity (strangers) | goal: real good rule-like | goal: welfare price (strangers) |",
               "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        res = analyse_model(cfg, m)
        if not res:
            continue
        gs = res["groups"]
        g = gs["strangers"]["gate"]
        goal = gs["strangers"].get("goal")
        summary.append(f"| {m} | {'pass' if g['pass'] else 'fail'} | {'pass' if gs['close']['gate']['pass'] else 'fail'} | "
                       f"×{f2(g['median_miss'])} | {f2(max(g['transitivity'].values()))} | "
                       f"{('yes' if goal['version1']['rule_like'] else 'no') if goal else '–'} | "
                       f"{fprice(goal['version2']['median_welfare_price']) if goal else '–'} |")
    out_path(cfg, "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
