"""Step 4: the main experiment.

Three sources of goal-seeking are varied independently, for every scenario:
  goal       the target is written into the system prompt (1) or not (0)
  monitored  the prompt says actions are logged and reviewed (1) or not (0)
  persona    no steering / toward the Assistant / away from it / a random
             direction of the same size (the control)
Each cell is answered samples_per_cell times in both option orders.

Before running for real, this script checks that prereg/PREREGISTRATION.md has
been locked. That is the point of the exercise: decide what would count as
evidence before seeing it.

Output: main.jsonl (one line per answer). Safe to stop and restart.
"""
import json
import re

from common import (ROOT, decision_system, decision_user, get_backend, load_config, load_direction, load_jsonl,
                    log, make_steer, out_path, parse_args, run_jobs)


def prereg_locked() -> bool:
    path = ROOT / "prereg" / "PREREGISTRATION.md"
    if not path.exists():
        return False
    m = re.search(r"^Locked on:(.*)$", path.read_text(encoding="utf-8"), flags=re.M)
    return bool(m) and "(not yet)" not in m.group(1) and m.group(1).strip() != ""


def main():
    args = parse_args(__doc__, extra=lambda ap: ap.add_argument(
        "--skip-prereg-check", action="store_true", help="run without a locked pre-registration"))
    cfg = load_config(args)
    if not (cfg["dry_run"] or cfg["quick_mode"] or args.skip_prereg_check or prereg_locked()):
        raise SystemExit("prereg/PREREGISTRATION.md is not locked yet. Fill it in, post or commit it so it has a\n"
                         "timestamp, then replace '(not yet)' on the 'Locked on:' line. (Or pass\n"
                         "--skip-prereg-check, and say so in the post.)")
    Mn = cfg["main"]
    direction = load_direction(cfg)

    alpha = Mn.get("alpha")
    if alpha is None:
        choice_path = out_path(cfg, "alpha_choice.json")
        if not choice_path.exists():
            raise SystemExit("Missing alpha_choice.json. Run 02_manipulation_check.py first.")
        alpha = json.loads(choice_path.read_text())["alpha"]
        if alpha is None and cfg["quick_mode"]:
            alpha = min(a for a in cfg["manipulation_check"]["alphas"] if a > 0)
            log(f"Quick mode: step 2 found no usable strength (normal for a tiny model); using {alpha} to test the code.")
        elif alpha is None:
            raise SystemExit("Step 2 found no usable steering strength. Fix that first (see its output).")
    log(f"Steering strength alpha = {alpha}")

    stakes_path = out_path(cfg, "stakes_choice.json")
    if not stakes_path.exists():
        raise SystemExit("Missing stakes_choice.json. Run 03_calibrate_stakes.py first.")
    stakes = json.loads(stakes_path.read_text())

    steers = {p: make_steer(direction, p, alpha) for p in Mn["persona_conditions"]}
    jobs = []
    for si, sc in enumerate(cfg["scenarios"]):
        if sc["id"] in stakes:
            level = int(stakes[sc["id"]]["level"])
        elif cfg["quick_mode"]:
            level = 1
        else:
            log(f"Skipping {sc['id']}: no calibrated harm level")
            continue
        for persona in Mn["persona_conditions"]:
            for goal in (0, 1):
                for mon in (0, 1):
                    for order in (0, 1):
                        for s in range(Mn["samples_per_cell"]):
                            jobs.append({"id": f"main|{sc['id']}|{persona}|{goal}|{mon}|{order}|{s}", "steer": persona,
                                         "system": decision_system(cfg, sc, goal, mon),
                                         "user": decision_user(cfg, sc, level, order),
                                         "meta": {"kind": "decision", "scenario": sc["id"], "scenario_idx": si,
                                                  "level": level, "goal": goal, "monitored": mon, "order": order}})
    path = out_path(cfg, "main.jsonl")
    run_jobs(get_backend(cfg), jobs, steers, path, cfg, cfg["generation"], label="main run")
    log(f"Done: {len(load_jsonl(path))} answers in {path}. Next: python 05_analyze.py")


if __name__ == "__main__":
    main()
