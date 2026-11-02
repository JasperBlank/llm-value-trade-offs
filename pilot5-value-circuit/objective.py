"""Stage `objective` (exploratory, CPU): pilot 4's goal-module question, read with the strong
last-position code. For each goal prompt against its twin without the goal line, at each layer in
config `layers`: the change in the readout of the target's value, of the partner's value, and of
their difference, against the change in the model's log-odds towards the target.
"Extra value": the value readouts move about as much as the choice. "Bypass": the choice moves,
the value readouts do not. Writes results/<run>/objective.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import acts_path, log, save_json


def run(ctx):
    cfg, rows = ctx.cfg, ctx.rows
    last = np.load(acts_path(cfg, "last.npy"), mmap_mode="r")
    meta = np.load(acts_path(cfg, "meta.npz"))
    D = np.load(Path(cfg["res"]) / "directions.npz")
    lo = meta["logodds"]
    idx_of = {r["id"]: r["index"] for r in rows}
    out = {}
    for version in ("goal1", "goal2"):
        G = [r for r in rows if r["group"] == "goal" and r["kind"] == version and r["id"] + "|twin" in idx_of]
        if not G:
            continue
        gi = np.array([r["index"] for r in G])
        wi = np.array([idx_of[r["id"] + "|twin"] for r in G])
        tslot = np.array([0 if r["key_a"] == r["goal_good"] else 1 for r in G])
        sign = np.where(tslot == 0, 1.0, -1.0)
        choice = float(np.mean((lo[gi] - lo[wi]) * sign))
        rec = {"n": int(len(G)), "choice_shift_toward_target": choice, "per_layer": {}}
        for L in cfg["layers"]:
            Xg = np.asarray(last[gi, L], np.float32)
            Xw = np.asarray(last[wi, L], np.float32)

            def ro(X, s):
                return (X - D[f"xm{s}_{L}"]) @ D[f"w{s}_{L}"] + float(D[f"ym{s}_{L}"])
            da, db = ro(Xg, "a") - ro(Xw, "a"), ro(Xg, "b") - ro(Xw, "b")
            t = np.where(tslot == 0, da, db)
            p = np.where(tslot == 0, db, da)
            rec["per_layer"][str(L)] = {"target_value_shift": float(t.mean()), "partner_value_shift": float(p.mean()),
                                        "difference_shift_toward_target": float((t - p).mean()),
                                        "fraction_of_choice_shift": float((t - p).mean() / choice) if choice else None}
            log(f"  {version}, layer {L}: choice {choice:+.2f} log-odds; target value {t.mean():+.2f}, "
                f"partner {p.mean():+.2f}, difference {(t - p).mean():+.2f}")
        out[version] = rec
    save_json(out, Path(cfg["res"]) / "objective.json")
    log("objective done")
