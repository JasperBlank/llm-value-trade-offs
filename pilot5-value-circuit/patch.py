"""Stage `patch` (exploratory): where does the decision reach the last position, and do the top
writers carry the good's value?

Pairs of fresh prompts that differ only in option A's good (same number of people, same B, same
wording): the recipient and its donor. At the LAST position only, the recipient's states are
replaced by the donor's:
  resid@L    the whole residual stream after block L-1, for L in config `patch_layers`
  top-k      the outputs of the k pieces (heads, MLPs before the primary layer) that carry most of
             the good's part of A's value (stage `circuit_analyze`), k in config `topk`
  random-k   k other pieces of the same blocks (control)
Transfer = slope (through zero) of the recipient's change in log-odds on the donor-minus-recipient
difference: 0 = nothing of the swapped good reached the decision through what was patched, 1 =
everything did. Writes results/<run>/patching.json and a figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import (Hooks, acts_path, batches, forward, layer_output, letter_ids, log, logodds, make_batch,
                    model_shape, replace_layer_output, save_json, values)


def transfer(obs, eff, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = float((obs * eff).sum() / (eff * eff).sum())
    bs = []
    for _ in range(reps):
        i = rng.integers(0, len(obs), len(obs))
        bs.append((obs[i] * eff[i]).sum() / (eff[i] * eff[i]).sum())
    return {"transfer": t, "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}


def run(ctx):
    import torch
    cfg, rows = ctx.cfg, ctx.rows
    tok, model, dev = ctx.model()
    encs = ctx.encs()
    sh = model_shape(model)
    H, hd = sh["H"], sh["hd"]
    Cc = cfg["circuit"]
    _, u = values()
    meta = np.load(acts_path(cfg, "meta.npz"))
    W = json.load(open(Path(cfg["res"]) / "writers.json"))
    by_id = {r["id"]: r for r in rows}
    lo0 = meta["logodds"]
    m = float(Cc["pair_max_abs_logodds"])
    pairs = [(by_id[r["recipient"]]["index"], r["index"]) for r in rows if r["group"] == "donor"
             and abs(lo0[by_id[r["recipient"]]["index"]]) <= m and abs(lo0[r["index"]]) <= m
             and lo0[r["index"]] != lo0[by_id[r["recipient"]]["index"]]]
    rng = np.random.default_rng(int(Cc["seed"]))
    if len(pairs) > Cc["n_pairs"]:
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), size=int(Cc["n_pairs"]), replace=False))]
    n = len(pairs)
    log(f"patch: {n} recipient/donor pairs")
    conds = [("resid", L) for L in Cc["patch_layers"]]
    conds += [("top", k) for k in Cc["topk"]] + [("random", k) for k in Cc["topk"]]
    sets = {("top", k): W["top"][:k] for k in Cc["topk"]}
    sets.update({("random", int(k)): v for k, v in W["random"].items()})
    lids = letter_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    layers = model.model.layers
    nL = len(layers)
    obs = np.zeros((len(conds), n))
    base_r, base_d = np.zeros(n), np.zeros(n)
    bs = int(cfg["batch_size"])
    for chunk in batches(n, bs):
        rec_idx = [pairs[j][0] for j in chunk]
        don_idx = [pairs[j][1] for j in chunk]
        # donor pass: keep its last-position states
        cache = {"z": {}, "mlp": {}, "blk": {}}
        with Hooks() as hk:
            for li, blk in enumerate(layers):
                hk.pre(blk.self_attn.o_proj, (lambda li: lambda mod, inp: cache["z"].__setitem__(li, inp[0][:, -1].clone()))(li))
                hk.fwd(blk.mlp, (lambda li: lambda mod, i, o: cache["mlp"].__setitem__(li, o[:, -1].clone()))(li))
                hk.fwd(blk, (lambda li: lambda mod, i, o: cache["blk"].__setitem__(li, layer_output(o)[:, -1].clone()))(li))
            ids, att, _ = make_batch(encs, don_idx, pad, torch, dev)
            base_d[chunk] = logodds(forward(model, ids, att, torch).logits[:, -1], lids, torch)
        ids, att, _ = make_batch(encs, rec_idx, pad, torch, dev)
        base_r[chunk] = logodds(forward(model, ids, att, torch).logits[:, -1], lids, torch)
        for ci, c in enumerate(conds):
            with Hooks() as hk:
                if c[0] == "resid":
                    L = c[1]
                    if L == 0:
                        continue

                    def h_blk(mod, i, o, L=L):
                        hh = layer_output(o).clone()
                        hh[:, -1] = cache["blk"][L - 1]
                        return replace_layer_output(o, hh)
                    hk.fwd(layers[L - 1], h_blk)
                else:
                    by_blk = {}
                    for blk_i, h in sets[c]:
                        by_blk.setdefault(blk_i, []).append(h)
                    for blk_i, hs in by_blk.items():
                        heads = [h for h in hs if h < H]
                        if heads:
                            def h_o(mod, inp, blk_i=blk_i, heads=heads):
                                z = inp[0].clone()
                                for h in heads:
                                    z[:, -1, h * hd:(h + 1) * hd] = cache["z"][blk_i][:, h * hd:(h + 1) * hd]
                                return (z,) + tuple(inp[1:])
                            hk.pre(layers[blk_i].self_attn.o_proj, h_o)
                        if H in hs:
                            def h_m(mod, i, o, blk_i=blk_i):
                                oo = o.clone()
                                oo[:, -1] = cache["mlp"][blk_i]
                                return oo
                            hk.fwd(layers[blk_i].mlp, h_m)
                obs[ci, chunk] = logodds(forward(model, ids, att, torch).logits[:, -1], lids, torch) - base_r[chunk]
        log(f"  {chunk[-1] + 1}/{n} pairs")
    eff = base_d - base_r
    pred = np.array([u[rows[d]["key_a"]] - u[rows[r]["key_a"]] for r, d in pairs])
    res = {"n_pairs": n, "effect_vs_predicted": transfer(eff, pred), "conditions": {}}
    for ci, c in enumerate(conds):
        res["conditions"][f"{c[0]}@{c[1]}" if c[0] == "resid" else f"{c[0]}-{c[1]}"] = transfer(obs[ci], eff)
    res["pieces"] = {f"top-{k}": W["top_names"][:k] for k in Cc["topk"]}
    for k, v in res["conditions"].items():
        log(f"  {k}: transfer {v['transfer']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
    log(f"  the model's own effect against the value model's prediction: slope {res['effect_vs_predicted']['transfer']:.2f}")
    save_json(res, Path(cfg["res"]) / "patching.json")
    np.savez(Path(cfg["res"]) / "patching.npz", obs=obs, eff=eff, pred=pred, pairs=np.array(pairs),
             conds=np.array([f"{a}@{b}" for a, b in conds]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    Ls = [L for L in Cc["patch_layers"]]
    t = [res["conditions"][f"resid@{L}"]["transfer"] for L in Ls]
    axes[0].plot(Ls, t, "o-")
    axes[0].axhline(1, color="grey", ls=":")
    axes[0].axhline(0, color="grey", lw=0.5)
    axes[0].set_xlabel("layer (whole residual stream patched at the last position)")
    axes[0].set_ylabel("transfer of the swapped good")
    axes[0].set_title("When has the decision reached the last position?", fontsize=9)
    ks = Cc["topk"]
    axes[1].plot(ks, [res["conditions"][f"top-{k}"]["transfer"] for k in ks], "o-", label="top writers")
    axes[1].plot(ks, [res["conditions"][f"random-{k}"]["transfer"] for k in ks], "s--", label="random pieces")
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("number of pieces patched")
    axes[1].set_title(f"Do the top writers carry the good's value? (pieces before layer {cfg['primary_layer']})", fontsize=9)
    axes[1].legend(fontsize=8)
    fig.savefig(Path(cfg["res"]) / "fig_patching.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log("patch done")
