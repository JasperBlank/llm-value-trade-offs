"""Score both prediction sets (predictions/predictions.yaml) against the results.

Run after analyze.py. Writes predictions_vs_results.md next to the results.
- Probabilities: the outcome (yes/no) and the Brier score (0 = perfect, 0.25 = a 50% guess).
- Numbers with a range: whether the outcome fell inside the 80% range, and the best guess.
- "Breaks first": the criterion that fails in the most model-and-group combinations
  (4B, 8B and 14B, strangers and family); a tie gives each tied criterion a share.
Unless a key says otherwise, outcomes are for Qwen3-14B, test goods, strangers.
"""
import math

import numpy as np

from common import ROOT, load_config, load_locked, load_yaml, out_path, parse_args

MODELS = ["4b", "8b", "14b"]
CRITERIA = ["1a", "1b", "2", "3"]


def load_json(path):
    import json
    return json.load(open(path)) if path.exists() else None


def outcomes(cfg, locked, A):
    """Every predicted quantity, from the analysis files. Missing ones are None."""
    a14 = A.get("14b")
    o = {"frame_B_selected": locked["frame"] == "B", "log_shape_selected": locked["shape"] == "log"}
    sc = locked.get("scores", {})
    f = locked["frame"]
    if f"{f}/log" in sc and f"{f}/linear" in sc:
        o["log_linear_tie"] = abs(sc[f"{f}/log"]["r"] - sc[f"{f}/linear"]["r"]) < 0.005
    o["gate_passes"] = {m: all(A[m]["groups"][b]["gate"]["pass"] for b in ("strangers", "close"))
                        for m in MODELS if m in A}
    fails = {c: sum(not A[m]["groups"][b]["gate"][c] for m in A if m in MODELS for b in ("strangers", "close"))
             for c in CRITERIA}
    top = max(fails.values()) if fails else 0
    o["first_criterion_to_break"] = {"fail_counts": fails,
                                     "most": [c for c, v in fails.items() if v == top] if top > 0 else []}
    if a14:
        g = a14["groups"]["strangers"]
        gate = g["gate"]
        o["borderline_14b"] = a14["borderline"]
        o["criterion2_r"] = gate["cv_r"]
        tf = list(gate["transitivity"].values())
        o["criterion3_factor"] = float("inf") if any(not np.isfinite(v) for v in tf) else max(tf)
        o["real_goods_with_slope"] = sum(np.isfinite(v["slope"]) and np.isfinite(v["se"]) and
                                         v["slope"] - cfg["gate"]["slope_z"] * v["se"] > 0
                                         for v in gate["slopes"].values())
        ex = g["extras"]
        o["menu_invariance_passes"] = ex["menu"]["pass"]
        o["description_invariance_passes"] = ex["description"]["pass"]
        o["loss_aversion"] = ex["loss_aversion"]["geometric_mean"]
        o["closeness_multiplier"] = a14["closeness"]["median"]
        o["closeness_uniform"] = a14["closeness"]["uniform"]
        if "goal" in g:
            v1, v2 = g["goal"]["version1"], g["goal"]["version2"]
            o["goal_v1_rule_like"] = v1["rule_like"]
            o["goal_v1_value_shift"] = v1["median_value_shift"]
            o["goal_v1_slope_ratio"] = float(np.nanmedian([p["partner_slope_ratio"] for p in v1["pairs"].values()]))
            o["goal_v2_rule_like"] = v2["rule_like"]
            o["goal_v2_welfare_price"] = v2["median_welfare_price"]
    return o


def num(x):
    """Prediction values may be numbers, 'above 8' or null."""
    if x is None:
        return None
    if isinstance(x, str):
        return float("inf") if "above" in x else float(x)
    return float(x)


def show(x):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if math.isinf(x):
            return "above 8 / undefined"
        if math.isnan(x):
            return "–"
        return f"{x:.2f}"
    return str(x)


def score_prob(p, outcome):
    if p is None or outcome is None:
        return None
    return (p - float(outcome)) ** 2


def score_range(pred, outcome):
    if not pred or outcome is None or (isinstance(outcome, float) and math.isnan(outcome)):
        return None
    lo, hi = (num(v) for v in pred.get("range", [None, None]))
    lo = -math.inf if lo is None else lo
    hi = math.inf if hi is None else hi
    return lo <= outcome <= hi


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    locked = load_locked(cfg, required=True)
    A = {m: load_json(out_path(cfg, f"analysis_{m}.json")) for m in MODELS}
    A = {m: v for m, v in A.items() if v}
    if not A:
        raise SystemExit("No analysis files yet. Run analyze.py first.")
    O = outcomes(cfg, locked, A)
    P = load_yaml(ROOT / "predictions" / "predictions.yaml")
    who = ["jasper", "claude"]
    md = ["# Pilot 2: predictions against results", "",
          f"Results: {'dry run (fake model)' if cfg['dry_run'] else ('quick test' if cfg['quick_mode'] else 'real run')}. "
          "Probabilities are scored with the Brier score (lower is better; always saying 50% scores 0.25). "
          "Ranges are 80% intervals.", "",
          "## Probabilities", "",
          "| prediction | outcome | Jasper | Brier | Claude | Brier |", "|---|---|---|---|---|---|"]
    briers = {w: [] for w in who}

    def prob_row(label, p_j, p_c, out):
        cells = []
        for w, p in (("jasper", p_j), ("claude", p_c)):
            b = score_prob(p, out)
            if b is not None:
                briers[w].append(b)
            cells += [f"{p:.0%}" if p is not None else "–", f"{b:.3f}" if b is not None else "–"]
        md.append(f"| {label} | {show(out)} | " + " | ".join(cells) + " |")

    for key, label in [("frame_B_selected", "frame B selected"), ("log_shape_selected", "log shape selected")]:
        prob_row(label, P[key].get("jasper"), P[key].get("claude"), O.get(key))
    prob_row("log and linear tied (Jasper only)", P["log_linear_tie"].get("jasper"), None, O.get("log_linear_tie"))
    for m in MODELS:
        prob_row(f"gate passes, {m}", P["gate_passes"][m].get("jasper"), P["gate_passes"][m].get("claude"),
                 O["gate_passes"].get(m))
    for key, label in [("borderline_14b", "32B rule triggers"),
                       ("menu_invariance_passes", "menu invariance passes"),
                       ("description_invariance_passes", "description invariance passes"),
                       ("closeness_uniform", "closeness is uniform"),
                       ("goal_v1_rule_like", "goal v1 (real good) rule-like"),
                       ("goal_v2_rule_like", "goal v2 (boxes) rule-like")]:
        prob_row(label, P[key].get("jasper"), P[key].get("claude"), O.get(key))
    md += ["", "| | Jasper | Claude |", "|---|---|---|",
           "| mean Brier score | " + " | ".join(f"{np.mean(briers[w]):.3f} ({len(briers[w])} predictions)"
                                                if briers[w] else "–" for w in who) + " |"]

    md += ["", "## Numbers", "", "| prediction | outcome | Jasper: best (range) | in range | Claude: best (range) | in range |",
           "|---|---|---|---|---|---|"]
    hits = {w: [] for w in who}
    for key, label in [("criterion2_r", "criterion 2, held-out r"),
                       ("criterion3_factor", "criterion 3, transitivity factor (worse triplet)"),
                       ("real_goods_with_slope", "real goods with a quantity slope (of 6)"),
                       ("loss_aversion", "loss aversion"),
                       ("closeness_multiplier", "closeness multiplier (family / strangers)"),
                       ("goal_v1_value_shift", "goal v1 value shift"),
                       ("goal_v1_slope_ratio", "goal v1 partner slope ratio"),
                       ("goal_v2_welfare_price", "goal v2 welfare price (people per box)")]:
        out = O.get(key)
        out = float(out) if out is not None else None
        cells = []
        for w in who:
            pred = P[key].get(w)
            if not pred:
                cells += ["–", "–"]
                continue
            r = pred.get("range", [None, None])
            hit = score_range(pred, out)
            if hit is not None:
                hits[w].append(hit)
            cells += [f"{show(num(pred.get('best')))} ({show(num(r[0]))} to {show(num(r[1]))})",
                      "–" if hit is None else ("yes" if hit else "no")]
        md.append(f"| {label} | {show(out)} | " + " | ".join(cells) + " |")
    md += ["", "| | Jasper | Claude |", "|---|---|---|",
           "| outcomes inside the 80% range | " + " | ".join(f"{sum(hits[w])} of {len(hits[w])}" for w in who) + " |"]

    fb = O["first_criterion_to_break"]
    md += ["", "## Which criterion breaks first", "",
           "Failures per criterion over the model-and-group combinations: " +
           ", ".join(f"{c}: {v}" for c, v in fb["fail_counts"].items()) + ".", ""]
    for w in who:
        pick = P["first_criterion_to_break"].get(w)
        if not fb["most"]:
            verdict = "no criterion failed"
        elif pick in fb["most"]:
            verdict = "hit" + (f" (shared with {len(fb['most']) - 1} other)" if len(fb["most"]) > 1 else "")
        else:
            verdict = "miss"
        md.append(f"- {w.capitalize()}: {pick} → {verdict}")

    text = "\n".join(md) + "\n"
    out_path(cfg, "predictions_vs_results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
