"""Score both prediction sets (predictions/predictions.yaml) against the results.

Run after analyze.py. Writes predictions_vs_results.md next to the results.
- Probabilities: the outcome (yes/no) and the Brier score (0 = perfect, 0.25 = a 50% guess).
- Numbers with a range: whether the outcome fell inside the 80% range.
- "Breaks first": the criterion failing in the most model-and-group combinations (14B, 8B, 32B
  x strangers, family); tied criteria share.
Unless a key says otherwise, outcomes are for Qwen3-14B, strangers.
"""
import json
import math

import numpy as np

from common import ROOT, load_config, load_yaml, out_path, parse_args

MODELS = ["14b", "8b", "32b"]
CRITERIA = ["1a", "1b", "2", "3"]
BEN = ("strangers", "close")


def outcomes(A, cfg):
    o = {"gate_passes": {m: all(A[m]["groups"][b]["gate"]["pass"] for b in BEN) for m in A}}
    fails = {c: sum(not A[m]["groups"][b]["gate"][c] for m in A for b in BEN) for c in CRITERIA}
    top = max(fails.values())
    o["first_criterion_to_break"] = {"fail_counts": fails, "most": [c for c, v in fails.items() if v == top] if top else []}
    a = A.get("14b")
    if not a:
        return o
    g = a["groups"]["strangers"]
    o["median_rate_miss"] = g["gate"]["median_miss"]
    o["pairs_within_x2"] = g["gate"]["pairs_within_x2"]
    o["transitivity_worse"] = max(g["gate"]["transitivity"].values())
    o["menu_invariance_passes"] = g["extras"]["menu"]["pass"]
    o["description_invariance_passes"] = g["extras"]["description"]["pass"]
    o["loss_aversion"] = g["extras"]["loss_aversion"]["geometric_mean"]
    o["loss_aversion_family"] = a["groups"]["close"]["extras"]["loss_aversion"]["geometric_mean"]
    o["closeness_multiplier"] = a["closeness"]["median"]
    o["closeness_uniform"] = a["closeness"]["uniform"]
    if "goal" in g:
        v1, v2 = g["goal"]["version1"], g["goal"]["version2"]
        o["goal_v1_rule_like"] = v1["rule_like"]
        for q in (32, 128):
            held = sum(p[f"p_target_1_vs_{q}"] >= cfg["goal_module"]["rule_min_prob"] for p in v1["pairs"].values())
            o[f"goal_v1_holds_at_1_vs_{q}"] = 3 * held >= 2 * len(v1["pairs"])
        o["goal_v2_rule_like"] = v2["rule_like"]
        o["goal_v2_welfare_price"] = v2["median_welfare_price"]
    return o


def num(x):
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
    if isinstance(x, (int, np.integer)):
        return str(x)
    if math.isinf(x):
        return "above 128 / infinite"
    if math.isnan(x):
        return "–"
    return f"{x:.2f}"


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    A = {}
    for m in MODELS:
        p = out_path(cfg, f"analysis_{m}.json")
        if p.exists():
            A[m] = json.load(open(p))
    if not A:
        raise SystemExit("No analysis files yet. Run analyze.py first.")
    O = outcomes(A, cfg)
    P = load_yaml(ROOT / "predictions" / "predictions.yaml")
    who = ["jasper", "claude"]
    md = ["# Pilot 3: predictions against results", "",
          f"Results: {'dry run (fake model)' if cfg['dry_run'] else 'real run'}. Brier score: lower is better; "
          "always saying 50% scores 0.25. Ranges are 80% intervals. Unless stated, 14B, strangers.", "",
          "## Probabilities", "", "| prediction | outcome | Jasper | Brier | Claude | Brier |", "|---|---|---|---|---|---|"]
    briers = {w: [] for w in who}

    def prob_row(label, preds, out):
        cells = []
        for w in who:
            p = preds.get(w) if preds else None
            b = (p - float(out)) ** 2 if p is not None and out is not None else None
            if b is not None:
                briers[w].append(b)
            cells += [f"{p:.0%}" if p is not None else "–", f"{b:.3f}" if b is not None else "–"]
        md.append(f"| {label} | {show(out)} | " + " | ".join(cells) + " |")

    for m in MODELS:
        prob_row(f"gate passes, {m}", P["gate_passes"].get(m), O["gate_passes"].get(m))
    for key, label in [("menu_invariance_passes", "menu invariance passes"),
                       ("description_invariance_passes", "description invariance passes"),
                       ("closeness_uniform", "closeness is uniform"),
                       ("goal_v1_rule_like", "goal v1 (real good) rule-like"),
                       ("goal_v1_holds_at_1_vs_32", "goal v1 override holds at 1 against 32"),
                       ("goal_v1_holds_at_1_vs_128", "goal v1 override holds at 1 against 128"),
                       ("goal_v2_rule_like", "goal v2 (files) rule-like")]:
        prob_row(label, P.get(key), O.get(key))
    md += ["", "| | Jasper | Claude |", "|---|---|---|",
           "| mean Brier score | " + " | ".join(f"{np.mean(briers[w]):.3f} ({len(briers[w])})" if briers[w] else "–"
                                                for w in who) + " |",
           "", "## Numbers", "", "| prediction | outcome | Jasper: best (range) | in range | Claude: best (range) | in range |",
           "|---|---|---|---|---|---|"]
    hits = {w: [] for w in who}
    for key, label in [("median_rate_miss", "criterion 2: median held-out rate miss"),
                       ("pairs_within_x2", "criterion 2: pairs within ×2 (of 15)"),
                       ("transitivity_worse", "criterion 3: transitivity factor, worse triplet"),
                       ("loss_aversion", "loss aversion"),
                       ("loss_aversion_family", "loss aversion, family"),
                       ("closeness_multiplier", "closeness multiplier"),
                       ("goal_v2_welfare_price", "goal v2 welfare price (people per file)")]:
        out = O.get(key)
        cells = []
        for w in who:
            pred = (P.get(key) or {}).get(w)
            if not pred:
                cells += ["–", "–"]
                continue
            lo, hi = (num(v) for v in pred.get("range", [None, None]))
            lo, hi = (-math.inf if lo is None else lo), (math.inf if hi is None else hi)
            hit = None if out is None or (isinstance(out, float) and math.isnan(out)) else bool(lo <= float(out) <= hi)
            if hit is not None:
                hits[w].append(hit)
            cells += [f"{show(num(pred.get('best')))} ({show(lo)} to {show(hi)})", "–" if hit is None else ("yes" if hit else "no")]
        md.append(f"| {label} | {show(out)} | " + " | ".join(cells) + " |")
    md += ["", "| | Jasper | Claude |", "|---|---|---|",
           "| outcomes inside the 80% range | " + " | ".join(f"{sum(hits[w])} of {len(hits[w])}" for w in who) + " |"]
    fb = O["first_criterion_to_break"]
    md += ["", "## Which criterion breaks first", "",
           "Failures per criterion: " + ", ".join(f"{c}: {v}" for c, v in fb["fail_counts"].items()) + ".", ""]
    for w in who:
        pick = (P.get("first_criterion_to_break") or {}).get(w)
        verdict = ("no criterion failed" if not fb["most"] else
                   ("hit" + (f" (shared with {len(fb['most']) - 1} other)" if len(fb["most"]) > 1 else ""))
                   if pick in fb["most"] else "miss")
        md.append(f"- {w.capitalize()}: {pick} → {verdict}")
    text = "\n".join(md) + "\n"
    out_path(cfg, "predictions_vs_results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
