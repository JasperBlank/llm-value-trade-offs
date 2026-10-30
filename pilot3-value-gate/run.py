"""Run pilot 3 on one model: stage 1 (scouting), the balance points, stage 2 (the test) and the
goal module, in that order.

  python run.py --model 14b

Every prompt is one forward pass; the letter probabilities are saved to
results/<real|quick|dryrun>/{scout,test,goal}_<model>.jsonl, and the balance points to
balance_<model>.json. Stopped or crashed? Run the same command again: it continues where it
left off.
"""
import numpy as np

from common import (GROUPS, get_backend, goal_jobs, load_config, load_jsonl, log, mirror_pairs, option_key,
                    out_path, pair_name, parse_args, real_pairs, run_jobs, save_json, scout_jobs, test_jobs)
from fit import cells, pair_rate


def balance_points(cfg, rows) -> dict:
    """For each pair and group: the ratio R at which the model is indifferent (units of the second
    good per unit of the first), from the order-averaged scouting cells. Clipped to 1:16..16:1."""
    lo, hi = cfg["scout"]["clip"]
    C = cells(rows, kinds=("scout",))
    out = {}
    for a, b, kind in real_pairs(cfg) + mirror_pairs(cfg):
        ka, kb = option_key(a), option_key(b)
        out[pair_name(a, b)] = {}
        for ben in GROUPS:
            R = pair_rate([c for c in C if c["beneficiary"] == ben], ka, kb)
            if not np.isfinite(R) and R != float("inf"):
                R = 1.0                        # no information: centre on 1:1
            out[pair_name(a, b)][ben] = float(min(max(R, lo), hi))
    return out


def main():
    def extra(ap):
        ap.add_argument("--model", required=True, help="a key under 'models' in config.yaml, e.g. 14b")
    args = parse_args(__doc__, extra)
    cfg = load_config(args)
    m = args.model
    if m not in cfg["models"] and not cfg["dry_run"]:
        raise SystemExit(f"Unknown model '{m}'. Choose from: {', '.join(cfg['models'])}")
    backend = get_backend(cfg, m)

    scout_path = out_path(cfg, f"scout_{m}.jsonl")
    run_jobs(backend, scout_jobs(cfg), scout_path, cfg, label=f"stage 1 (scouting) {m}")
    balance = balance_points(cfg, load_jsonl(scout_path))
    save_json(balance, out_path(cfg, f"balance_{m}.json"))
    log(f"Balance points saved: balance_{m}.json")

    run_jobs(backend, test_jobs(cfg, balance), out_path(cfg, f"test_{m}.jsonl"), cfg, label=f"stage 2 (test) {m}")
    run_jobs(backend, goal_jobs(cfg), out_path(cfg, f"goal_{m}.jsonl"), cfg, label=f"goal module {m}")
    log(f"Done: {m}")


if __name__ == "__main__":
    main()
