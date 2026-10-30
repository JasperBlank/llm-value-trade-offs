"""Step 2: does steering actually change the persona, and how hard can we push?

For each steering strength (alpha) and each condition (toward / away / random) we:
  - ask identity questions ("who are you?") and count how often the model says
    it is an AI: steering AWAY from the Assistant should lower this;
  - ask two decision questions and count how often the answer still has a
    readable "DECISION: A/B" line, and how often answers are coherent at all.

Pre-registered rule for choosing the strength used in the main run:
  the LARGEST alpha at which toward, away and random ALL keep at least
  min_parse_rate readable decisions and min_coherence coherent answers.
The manipulation counts as working if, at that alpha, "away" lowers AI
self-identification by at least min_identity_shift compared with no steering.

Outputs: manipulation_check.jsonl, alpha_choice.json, manipulation_check.md
"""
from collections import defaultdict

from common import (decision_system, decision_user, get_backend, is_coherent, load_config,
                    load_direction, load_jsonl, log, make_steer, out_path, parse_args, parse_decision,
                    run_jobs, save_json, says_ai)


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    M, P = cfg["manipulation_check"], cfg["prompts"]
    direction = load_direction(cfg)
    backend = get_backend(cfg)

    conditions = [("none", 0.0)] + [(p, a) for a in M["alphas"] if a > 0 for p in ("toward", "away", "random")]
    steers = {f"{p}@{a}": make_steer(direction, p, a) for p, a in conditions}

    jobs = []
    for p, a in conditions:
        key = f"{p}@{a}"
        for qi, q in enumerate(P["identity_prompts"]):
            for s in range(M["identity_samples"]):
                jobs.append({"id": f"id|{key}|{qi}|{s}", "steer": key, "system": None, "user": q,
                             "meta": {"kind": "identity", "q": qi}})
        for si, sc in enumerate(cfg["scenarios"][: M["decision_scenarios"]]):
            for order in (0, 1):
                for s in range(M["decision_samples"]):
                    jobs.append({"id": f"dec|{key}|{sc['id']}|{order}|{s}", "steer": key,
                                 "system": decision_system(cfg, sc, 1, 1),
                                 "user": decision_user(cfg, sc, 1, order),
                                 "meta": {"kind": "decision", "scenario": sc["id"], "scenario_idx": si,
                                          "level": 1, "goal": 1, "monitored": 1, "order": order}})
    path = out_path(cfg, "manipulation_check.jsonl")
    run_jobs(backend, jobs, steers, path, cfg, cfg["generation"], label="manipulation check")

    stats = defaultdict(lambda: defaultdict(list))
    examples = defaultdict(list)
    for r in load_jsonl(path):
        key = r["steer"]
        stats[key]["coherent"].append(is_coherent(r["text"]))
        if r["kind"] == "identity":
            stats[key]["ai"].append(says_ai(r["text"]))
            if len(examples[key]) < 3:
                examples[key].append(r["text"].replace("\n", " ")[:220])
        else:
            stats[key]["parsed"].append(parse_decision(r["text"]) is not None)

    def rate(key, what):
        xs = stats[key][what]
        return sum(xs) / len(xs) if xs else float("nan")

    table = []
    for p, a in conditions:
        key = f"{p}@{a}"
        table.append({"condition": p, "alpha": a, "says_ai": rate(key, "ai"),
                      "readable_decision": rate(key, "parsed"), "coherent": rate(key, "coherent")})

    ok = []
    for a in sorted({a for _, a in conditions if a > 0}):
        rows = [t for t in table if t["alpha"] == a]
        if all(t["readable_decision"] >= M["min_parse_rate"] and t["coherent"] >= M["min_coherence"] for t in rows):
            ok.append(a)
    chosen = max(ok) if ok else None
    base_ai = rate("none@0.0", "ai")
    shift = base_ai - rate(f"away@{chosen}", "ai") if chosen is not None else float("nan")
    works = chosen is not None and shift >= M["min_identity_shift"]

    result = {"alpha": chosen, "identity_shift_away": shift, "says_ai_unsteered": base_ai,
              "manipulation_works": works, "table": table}
    save_json(result, out_path(cfg, "alpha_choice.json"))

    md = ["| condition | alpha | says it is an AI | readable decision | coherent |", "|---|---|---|---|---|"]
    for t in table:
        md.append(f"| {t['condition']} | {t['alpha']} | {t['says_ai']:.0%} | {t['readable_decision']:.0%} "
                  f"| {t['coherent']:.0%} |")
    md += ["", "Example identity answers:", ""]
    for p, a in conditions:
        for e in examples[f"{p}@{a}"]:
            md.append(f"- **{p} @ {a}**: {e}")
    out_path(cfg, "manipulation_check.md").write_text("\n".join(md), encoding="utf-8")

    print("\n" + "\n".join(md[: len(table) + 2]) + "\n")
    if chosen is None:
        log("No strength passed the readability rule. Lower the alphas in config.yaml and rerun this step.")
    else:
        log(f"Chosen strength: alpha = {chosen}. Away steering lowers AI self-identification by "
            f"{shift:.0%} (rule: at least {M['min_identity_shift']:.0%}).")
        if not works:
            log("WARNING: the manipulation is weak. Report this; a null persona effect would then "
                "mean little. Consider another layer (config 'layer') and rerun steps 1-2.")
    log(f"Details and example answers: {out_path(cfg, 'manipulation_check.md')}")


if __name__ == "__main__":
    main()
