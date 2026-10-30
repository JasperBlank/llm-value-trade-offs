"""Step 7: score the pre-registered predictions against the results.

Reads prereg/predictions.yaml and the outputs of steps 2, 3 (including the dose-response
check, 10), 5 (including 9) and 6, and writes
predictions_vs_results.md:

  - for each effect: the result, and whether each person's best guess was close and
    whether the result fell inside their 80% range;
  - for each yes/no question: what happened, and the probability each person gave it;
  - a Brier score per person over the yes/no questions (0 is perfect, 0.25 is what you
    get by always saying 50%, lower is better).

If someone's 80% ranges are well calibrated, about 8 in 10 results land inside them.
"""
import json

from common import ROOT, load_config, load_yaml, log, out_path, parse_args

EFFECT_LABELS = {
    "goal": "Goal in prompt", "unmonitored": "Unmonitored", "away": "Away from Assistant",
    "toward": "Toward Assistant", "random": "Random push", "individual_vs_group": "Named individual vs group",
}


def read_json(cfg, name):
    p = out_path(cfg, name)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    preds = load_yaml(ROOT / "prereg" / "predictions.yaml")
    people = list(preds)
    summary = read_json(cfg, "summary.json")
    if summary is None:
        if cfg["quick_mode"]:
            log("Quick mode: nothing to compare (no analysable results from the small model).")
            return
        raise SystemExit("Missing summary.json. Run 05_analyze.py first.")
    alpha = read_json(cfg, "alpha_choice.json") or {}
    stakes = read_json(cfg, "stakes_choice.json") or {}
    harm = read_json(cfg, "harm_directions.json") or {}

    results = {k: v["estimate"] for k, v in summary["effects_relative_pct"].items()}
    if summary.get("individual_vs_group"):
        results["individual_vs_group"] = summary["individual_vs_group"]["rel"]

    md = ["# Predictions versus results", "",
          "Effects in relative change of the care rate. ✓ = the result fell inside that person's 80% range.", "",
          "| effect | result | " + " | ".join(f"{p} (best, range)" for p in people) + " |",
          "|---|---|" + "---|" * len(people)]
    inside = {p: [] for p in people}
    for k, label in EFFECT_LABELS.items():
        if k not in results:
            continue
        r = results[k]
        cells = []
        for p in people:
            q = preds[p].get(k)
            if not q:
                cells.append("–")
                continue
            ok = q["low"] <= r <= q["high"]
            inside[p].append(ok)
            cells.append(f"{q['best']:+d}% ({q['low']:+d} to {q['high']:+d}) {'✓' if ok else '✗'}")
        md.append(f"| {label} | {r:+.0f}% | " + " | ".join(cells) + " |")

    dose = read_json(cfg, "dose_response.json") or {}
    counts = {
        "stuck_versions": ("Versions stuck at floor or ceiling",
                           sum(1 for v in stakes.values() if v.get("note") != "ok") if stakes else None, "{}"),
        "rising_ladders": ("10. Harm ladders where care rises with the harm", dose.get("rising_ladders"), "{}"),
        "mean_rise_pp": ("10b. Average rise from mildest to most serious harm (points)", dose.get("mean_rise_pp"),
                         "{:+.1f}"),
    }
    for key, (label, value, fmt) in counts.items():
        if value is None:
            continue
        cells = []
        for p in people:
            q = preds[p].get(key)
            if not q or q.get("best") is None:
                cells.append("–")
                continue
            if q.get("low") is None or q.get("high") is None:
                cells.append(f"{q['best']} (no range given, not scored)")
                continue
            ok = q["low"] <= value <= q["high"]
            inside[p].append(ok)
            cells.append(f"{q['best']} ({q['low']} to {q['high']}) {'✓' if ok else '✗'}")
        md.append(f"| {label} | {fmt.format(value)} | " + " | ".join(cells) + " |")

    md += ["", "| | " + " | ".join(people) + " |", "|---|" + "---|" * len(people),
           "| Results inside the 80% range | " + " | ".join(
               f"{sum(v)} of {len(v)}" for v in inside.values()) + " |", ""]

    # Yes/no questions and the largest-effect question.
    events = {
        "manipulation_works": ("The steering manipulation worked as pre-specified", alpha.get("manipulation_works")),
        "same_type_more_alike": ("Same harm type more alike inside the network (6a)", harm.get("same_type_more_alike")),
        "one_shared_signal": ("One shared harm signal (6b)", harm.get("one_shared_signal")),
        "persona_profile_by_harm_type": ("Steering away moves harm types differently (9)",
                                         (summary.get("persona_profile") or {}).get("supported")),
    }
    md += ["| question | what happened | " + " | ".join(f"{p}: probability given" for p in people) + " |",
           "|---|---|" + "---|" * len(people)]
    brier = {p: [] for p in people}
    for k, (label, outcome) in events.items():
        if outcome is None:
            continue
        cells = []
        for p in people:
            prob = preds[p].get(k)
            if prob is None:
                cells.append("–")
                continue
            brier[p].append((prob - float(outcome)) ** 2)
            cells.append(f"{prob:.0%}")
        md.append(f"| {label} | {'yes' if outcome else 'no'} | " + " | ".join(cells) + " |")
    winner = summary.get("largest_effect")
    if winner:
        cells = []
        for p in people:
            dist = preds[p].get("largest_effect", {})
            given = dist.get(winner, 0.0)
            brier[p].append(sum((dist.get(k, 0.0) - (1.0 if k == winner else 0.0)) ** 2 for k in
                                ("goal", "unmonitored", "away")) / 3)
            cells.append(f"{given:.0%} on {winner}")
        md.append(f"| Largest effect | {winner} | " + " | ".join(cells) + " |")
    md += ["", "| | " + " | ".join(people) + " |", "|---|" + "---|" * len(people),
           "| Brier score (lower is better; 0.25 = always saying 50%) | " + " | ".join(
               f"{sum(v) / len(v):.3f} over {len(v)}" if v else "–" for v in brier.values()) + " |"]

    path = out_path(cfg, "predictions_vs_results.md")
    path.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    log(f"Wrote {path}")


if __name__ == "__main__":
    main()
