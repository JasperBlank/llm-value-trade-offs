"""Stage `readers` (exploratory): when A's value is pushed up, do the heads that write the answer
look more at the "A" label?

On the first `reader_steer_prompts` of the steering prompts, A's value direction at the primary
layer is pushed at -2, 0, +2 (the steering test's setting), and the attention of the top reader
heads (stage `circuit_analyze`) from the last position to the "A" and "B" labels is recorded.
Heads in blocks before the push cannot be affected (reported as such).
Writes results/<run>/readers.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import REGIONS, Hooks, batches, forward, letter_ids, log, logodds, make_batch, save_json
from steer import push_last


def run(ctx):
    import torch
    cfg, rows = ctx.cfg, ctx.rows
    tok, model, dev = ctx.model()
    encs = ctx.encs()
    P = int(cfg["primary_layer"])
    circ = json.load(open(Path(cfg["res"]) / "circuit.json"))
    st = json.load(open(Path(cfg["res"]) / "steering.json"))
    D = np.load(Path(cfg["res"]) / "directions.npz")
    by_id = {r["id"]: r["index"] for r in rows}
    idx = [by_id[i] for i in st["prompt_ids"][:int(cfg["circuit"]["reader_steer_prompts"])]]
    heads = [(h["block"], h["head"]) for h in circ["readers"]["top_heads"]]
    ia, ib = REGIONS.index("a_label"), REGIONS.index("b_label")
    lids = letter_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    vec = torch.tensor(D[f"steer_a_{P}"], dtype=torch.float32, device=dev)
    res = {"n_prompts": len(idx), "layer": P, "heads": {}, "logodds_shift": {}}
    att = {s: {hh: [] for hh in heads} for s in (-2.0, 0.0, 2.0)}
    los = {s: [] for s in att}
    blocks = sorted({b for b, _ in heads})
    for s in att:
        state = {"scale": s}
        for chunk in batches(len(idx), int(cfg["batch_size"])):
            ii = [idx[j] for j in chunk]
            ids, am, regs = make_batch(encs, ii, pad, torch, dev)
            cap = {}
            with Hooks() as hk:
                hk.fwd(*push_last(model, P, vec, state))
                for b in blocks:
                    hk.fwd(model.model.layers[b].self_attn,
                           (lambda b: lambda m, i, o: cap.__setitem__(b, o[1][:, :, -1, :].float().cpu().numpy()))(b))
                o = forward(model, ids, am, torch)
            los[s].append(logodds(o.logits[:, -1], lids, torch))
            for b, h in heads:
                a = cap[b][:, h]                                            # (B, T)
                att[s][(b, h)].append(np.stack([(a * (regs == ia)).sum(1), (a * (regs == ib)).sum(1)], 1))
    for s in att:
        los[s] = np.concatenate(los[s])
    for b, h in heads:
        A = {s: np.concatenate(att[s][(b, h)]) for s in att}
        d = {s: (A[s][:, 0] - A[s][:, 1]) for s in A}
        res["heads"][f"head {b}.{h}"] = {
            "after_push": b >= P,
            "attention_a_label": {str(s): float(A[s][:, 0].mean()) for s in A},
            "attention_b_label": {str(s): float(A[s][:, 1].mean()) for s in A},
            "a_minus_b_change_per_unit": float(((d[2.0] - d[-2.0]) / 4).mean())}
    res["logodds_shift"] = {str(s): float((los[s] - los[0.0]).mean()) for s in los}
    for k, v in res["heads"].items():
        log(f"  {k}: attention to A label {v['attention_a_label']['-2.0']:.2f} -> {v['attention_a_label']['2.0']:.2f}, "
            f"to B label {v['attention_b_label']['-2.0']:.2f} -> {v['attention_b_label']['2.0']:.2f}"
            + ("" if v["after_push"] else " (before the push)"))
    save_json(res, Path(cfg["res"]) / "readers.json")
    log("readers done")
