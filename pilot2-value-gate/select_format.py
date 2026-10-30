"""After the tune phase: lock the format by the pre-registered rule (DESIGN.md, section 3).

1. A frame qualifies only if, for every tune model, at least 95% of the answer probability
   falls on "A" or "B" (mean over all two-option prompts).
2. Among qualifying frames and the two value-function shapes (log, linear), choose the
   combination with the best held-out correlation of the utility model on the tune goods
   (real goods, within- and across-valence pairs), averaged over the tune models and both
   beneficiary groups.
3. Ties (a difference under 0.005) go to the smaller transitivity error.

It also flags within-valence domain pairs whose indifference point fell outside the grid
(1 unit of one good worth more than 8 of the other), so the matching test pair gets the
wider grid.

Writes locked_format.json and tune_report.md. Commit both to GitHub before the test phase.
"""
import datetime as dt
import itertools

import numpy as np

from common import goods_in, load_config, load_jsonl, log, out_path, parse_args, save_json
from fit import cv_correlation, pair_rate, transitivity_factor, two_option_table


def recs_for(T, frame, ben, kinds=("within", "cross")):
    return [r for r in T if r["kind"] in kinds and r["beneficiary"] == ben and r["frame_"] == frame]


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    models = cfg["tune_models"]
    trios = [goods_in(cfg, "tune", "gain"), goods_in(cfg, "tune", "harm_prevented")]
    tables = {}
    for m in models:
        rows = load_jsonl(out_path(cfg, f"tune_{m}.jsonl"))
        if not rows:
            raise SystemExit(f"No tune results for model {m}. Run: python run.py --phase tune --model {m}")
        tables[m] = two_option_table(rows)

    compliance = {f: {m: float(np.mean([t["mass"] for t in tables[m] if t["frame_"] == f])) for m in models}
                  for f in ("A", "B")}
    qualifying = [f for f in ("A", "B") if all(v >= cfg["compliance"]["min_frame"] for v in compliance[f].values())]
    log(f"Probability on the letters A/B: {compliance}. Qualifying frames: {qualifying or 'none'}")
    if not qualifying:
        raise SystemExit("No frame puts at least 95% of its probability on A/B. Look at a few prompts before going on.")

    scores = {}
    for f in qualifying:
        for shape in ("log", "linear"):
            rs, ts = [], []
            for m in models:
                for ben in ("strangers", "close"):
                    recs = recs_for(tables[m], f, ben)
                    rs.append(cv_correlation(recs, shape)[0])
                    ts += [transitivity_factor(recs, [g['id'] for g in trio], shape) for trio in trios]
            scores[(f, shape)] = {"r": float(np.nanmean(rs)), "transitivity": float(np.nanmean(ts))}
            log(f"  frame {f}, {shape}: held-out r = {scores[(f, shape)]['r']:.3f}, "
                f"transitivity factor = {scores[(f, shape)]['transitivity']:.2f}")
    best_r = max(v["r"] for v in scores.values())
    tied = [k for k, v in scores.items() if best_r - v["r"] < 0.005]
    frame, shape = min(tied, key=lambda k: scores[k]["transitivity"])
    log(f"Chosen: frame {frame}, {shape} value function")

    # Domain pairs whose indifference point lies outside 1:8..8:1 in the tune phase.
    wide = set()
    for m in models:
        for ben in ("strangers", "close"):
            recs = recs_for(tables[m], frame, ben, kinds=("within",))
            for trio in trios:
                for a, b in itertools.combinations(trio, 2):
                    rate = pair_rate(recs, a["id"], b["id"], shape)
                    if np.isfinite(rate) and (rate > 8 or rate < 1 / 8):
                        wide.add((a["valence"],) + tuple(sorted([a["domain"], b["domain"]])))
    locked = {"frame": frame, "shape": shape, "wide_pairs": sorted(map(list, wide)),
              "compliance": compliance, "scores": {f"{k[0]}/{k[1]}": v for k, v in scores.items()},
              "decided_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
              "rule": "DESIGN.md section 3; applied by select_format.py"}
    save_json(locked, out_path(cfg, "locked_format.json"))

    md = ["# Tune phase: format selection", "",
          f"Tune models: {', '.join(models)}. Decided {locked['decided_at']} by the pre-registered rule.", "",
          "| frame | " + " | ".join(f"probability on A/B ({m})" for m in models) + " |",
          "|---|" + "---|" * len(models)]
    md += [f"| {f} | " + " | ".join(f"{compliance[f][m]:.1%}" for m in models) + " |" for f in ("A", "B")]
    md += ["", "| frame and shape | held-out r | transitivity factor |", "|---|---|---|"]
    md += [f"| {k[0]}, {k[1]} | {v['r']:.3f} | {v['transitivity']:.2f} |" for k, v in scores.items()]
    md += ["", f"**Chosen: frame {frame}, {shape} value function.**",
           f"Widened grids for test pairs (valence, domains): {sorted(wide) or 'none'}"]
    out_path(cfg, "tune_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
