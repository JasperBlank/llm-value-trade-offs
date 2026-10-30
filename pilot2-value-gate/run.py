"""Run one phase of pilot 2 on one model.

  python run.py --phase tune --model 8b    tune goods, both frames (A and B)
  python run.py --phase test --model 14b   test goods, the locked frame: the gate and the extras
  python run.py --phase goal --model 14b   test goods, the locked frame: the goal module

Every prompt is one forward pass; the answer probabilities for the letters are saved to
results/<real|quick|dryrun>/<phase>_<model>.jsonl. Stopped or crashed? Run the same
command again and it continues where it left off.
"""
from common import build_jobs, get_backend, load_config, load_locked, log, out_path, parse_args, run_jobs


def main():
    def extra(ap):
        ap.add_argument("--phase", required=True, choices=["tune", "test", "goal"])
        ap.add_argument("--model", required=True, help="a key under 'models' in config.yaml, e.g. 8b")
    args = parse_args(__doc__, extra)
    cfg = load_config(args)
    if args.model not in cfg["models"]:
        raise SystemExit(f"Unknown model '{args.model}'. Choose from: {', '.join(cfg['models'])}")
    locked = None if args.phase == "tune" else load_locked(cfg, required=True)
    frames = ["A", "B"] if args.phase == "tune" else [locked["frame"]]
    if locked:
        log(f"Locked format: frame {locked['frame']}, {locked['shape']} value function, "
            f"widened grids: {locked.get('wide_pairs') or 'none'}")
    backend = get_backend(cfg, args.model)
    path = out_path(cfg, f"{args.phase}_{args.model}.jsonl")
    for frame in frames:
        jobs = build_jobs(cfg, args.phase, frame, locked)
        run_jobs(backend, jobs, path, cfg, label=f"{args.phase} {args.model} frame {frame}")
    log(f"Done: {path}")


if __name__ == "__main__":
    main()
