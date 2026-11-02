"""Stage `steer`: THE PRE-REGISTERED TEST (DESIGN.md, section 3).

On fresh prompts (three new wordings) the model is undecided about, a direction is added to the
residual stream at the LAST position only, at the output of block L-1 (so the probe at layer L
reads it), for L in config `layers`:
  A's value direction at strengths -2, -1, +1, +2 (value units)   predicted slope +1
  B's value direction, same strengths                               predicted slope -1
  random direction and two shuffled-values directions (same length as A's), at -2 and +2
Measured: the change in the log-odds of answering A. Per prompt, the slope = sum(s * shift) / sum(s^2);
reported as the mean over prompts with a bootstrap 95% interval.
Writes results/<run>/steering.json and steering.npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import (Hooks, acts_path, batches, bootstrap_mean_ci, forward, layer_output, letter_ids, log, logodds,
                    make_batch, replace_layer_output, save_json)

MONEY = {"G_energy", "H_fine", "H_fine~gainword"}


def select_prompts(cfg, rows, meta):
    S = cfg["steering"]
    fresh = [r for r in rows if r["group"] == "fresh"]
    pa = meta["p_a"] / np.maximum(meta["p_a"] + meta["p_b"], 1e-12)
    lo, hi = S["window"]
    near = [r["index"] for r in fresh if lo < pa[r["index"]] < hi]
    rng = np.random.default_rng(int(S["seed"]))
    if len(near) > S["max_prompts"]:
        near = sorted(rng.choice(near, size=int(S["max_prompts"]), replace=False).tolist())
    if len(near) < S["min_prompts"]:
        near = sorted(sorted((r["index"] for r in fresh), key=lambda i: abs(pa[i] - 0.5))[:int(S["min_prompts"])])
    return near


def conditions(cfg, D):
    S = cfg["steering"]
    out = []
    for L in cfg["layers"]:
        for name in ("a", "b"):
            for s in S["strengths"]:
                if s != 0:
                    out.append({"dir": name, "layer": L, "strength": float(s), "vec": D[f"steer_{name}_{L}"]})
        for name in ["random"] + [f"shuffled{j + 1}" for j in range(int(S["shuffles"]))]:
            for s in S["control_strengths"]:
                out.append({"dir": name, "layer": L, "strength": float(s), "vec": D[f"steer_{name}_{L}"]})
    return out


def push_last(model, L, vec_t, state):
    """Forward hook on block L-1: adds state['scale'] * vec to the last position."""
    def hook(module, inputs, output):
        if state["scale"] == 0:
            return output
        h = layer_output(output).clone()
        h[:, -1] = (h[:, -1].float() + state["scale"] * vec_t).to(h.dtype)
        return replace_layer_output(output, h)
    return model.model.layers[L - 1], hook


def slopes(shift_by_s: dict) -> np.ndarray:
    ss = np.array(sorted(shift_by_s))
    Y = np.stack([shift_by_s[s] for s in ss], 1)                    # (prompts, strengths)
    return (Y * ss).sum(1) / (ss ** 2).sum()


def run(ctx):
    import torch
    cfg, rows = ctx.cfg, ctx.rows
    tok, model, dev = ctx.model()
    encs = ctx.encs()
    S, C = cfg["steering"], cfg["steering"]["criteria"]
    D = dict(np.load(Path(cfg["res"]) / "directions.npz"))
    meta = np.load(acts_path(cfg, "meta.npz"))
    idx = select_prompts(cfg, rows, meta)
    n = len(idx)
    lids = letter_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = sorted(range(n), key=lambda j: len(encs[idx[j]]["ids"]))
    bs = int(cfg["batch_size"])
    log(f"steer: {n} fresh prompts with {S['window'][0]} < P(A) < {S['window'][1]}")

    def run_all(state=None):
        out = np.zeros(n)
        for chunk in batches(n, bs):
            jj = [order[j] for j in chunk]
            ids, att, _ = make_batch(encs, [idx[j] for j in jj], pad, torch, dev)
            o = forward(model, ids, att, torch)
            out[jj] = logodds(o.logits[:, -1], lids, torch)
        return out

    base = run_all()
    conds = conditions(cfg, D)
    shifts = np.zeros((len(conds), n))
    for ci, c in enumerate(conds):
        state = {"scale": c["strength"]}
        vec = torch.tensor(c["vec"], dtype=torch.float32, device=dev)
        with Hooks() as hk:
            hk.fwd(*push_last(model, c["layer"], vec, state))
            shifts[ci] = run_all() - base
        if ci % 8 == 0 or ci == len(conds) - 1:
            log(f"  {ci + 1}/{len(conds)}: {c['dir']} at layer {c['layer']}, x{c['strength']:+g}: "
                f"mean shift {shifts[ci].mean():+.3f}")

    keys_a = [rows[i]["key_a"] for i in idx]
    groups = {"gain_worded": np.array([k.startswith("G_") or "~gainword" in k for k in keys_a]),
              "harm_worded": np.array([k.startswith("H_") and "~" not in k for k in keys_a]),
              "money": np.array([k in MONEY for k in keys_a]),
              "nonmoney": np.array([k not in MONEY for k in keys_a])}
    res = {"n_prompts": n, "prompt_ids": [rows[i]["id"] for i in idx], "base_logodds_mean": float(base.mean()),
           "per_layer": {}, "criteria": C, "primary_layer": cfg["primary_layer"]}
    per_prompt = {}
    rng = np.random.default_rng(int(S["seed"]))
    for L in cfg["layers"]:
        lay = {}
        for name in ["a", "b", "random"] + [f"shuffled{j + 1}" for j in range(int(S["shuffles"]))]:
            by_s = {c["strength"]: shifts[ci] for ci, c in enumerate(conds) if c["layer"] == L and c["dir"] == name}
            sl = slopes(by_s)
            per_prompt[f"{name}_{L}"] = sl
            m, lo_, hi_ = bootstrap_mean_ci(sl, reps=int(S["bootstrap"]), seed=int(S["seed"]))
            lay[name] = {"slope": m, "ci95": [lo_, hi_],
                         "mean_shift": {str(s): float(v.mean()) for s, v in sorted(by_s.items())}}
        # one currency: A's slope for different kinds of A's good (ratio, bootstrap within groups)
        sa = per_prompt[f"a_{L}"]
        cur = {}
        for g1, g2 in (("gain_worded", "harm_worded"), ("money", "nonmoney")):
            x1, x2 = sa[groups[g1]], sa[groups[g2]]
            if len(x1) < 5 or len(x2) < 5:
                cur[f"{g1}_vs_{g2}"] = {"n": [int(len(x1)), int(len(x2))], "ratio": float("nan")}
                continue
            boots = [x1[rng.integers(0, len(x1), len(x1))].mean() / x2[rng.integers(0, len(x2), len(x2))].mean()
                     for _ in range(int(S["bootstrap"]))]
            cur[f"{g1}_vs_{g2}"] = {"n": [int(len(x1)), int(len(x2))], "slopes": [float(x1.mean()), float(x2.mean())],
                                    "ratio": float(x1.mean() / x2.mean()),
                                    "ratio_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
        lay["currency"] = cur
        res["per_layer"][str(L)] = lay
        log(f"  layer {L}: slope A {lay['a']['slope']:+.3f} [{lay['a']['ci95'][0]:+.3f}, {lay['a']['ci95'][1]:+.3f}], "
            f"B {lay['b']['slope']:+.3f}, random {lay['random']['slope']:+.3f}, "
            + ", ".join(f"shuffled{j + 1} {lay[f'shuffled{j + 1}']['slope']:+.3f}" for j in range(int(S["shuffles"]))))
    np.savez(Path(cfg["res"]) / "steering.npz", base=base, shifts=shifts, prompt_index=np.array(idx),
             cond_dir=np.array([c["dir"] for c in conds]), cond_layer=np.array([c["layer"] for c in conds]),
             cond_strength=np.array([c["strength"] for c in conds]), **{f"slope_{k}": v for k, v in per_prompt.items()})
    save_json(res, Path(cfg["res"]) / "steering.json")
    log("steer done")
