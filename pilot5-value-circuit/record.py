"""Stage `record`: one pass over every prompt; keeps the last-position residual stream at every layer.

Writes to acts_dir:
  last.npy   (prompts, layers+1, width) float16: the residual stream at the last token (where the
             answer is read). Layer 0 = embeddings, layer L = output of block L (the last one is
             after the final norm, as in pilot 4).
  meta.npz   P(A), P(B) (summed over spellings), log-odds, prompt ids.
If pilot 4's recordings are still on the volume, a few prompts are compared with them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import acts_path, batches, forward, letter_ids, log, make_batch, model_shape, p_letters, save_json


def run(ctx):
    import torch
    cfg, rows = ctx.cfg, ctx.rows
    tok, model, dev = ctx.model()
    encs = ctx.encs()
    sh = model_shape(model)
    N, L, d = len(rows), sh["n_layers"] + 1, sh["d"]
    lids = letter_ids(tok)
    log(f"record: {N} prompts, {sh['n_layers']} layers, width {d}, letter tokens {lids}")
    last = np.lib.format.open_memmap(acts_path(cfg, "last.npy"), mode="w+", dtype=np.float16, shape=(N, L, d))
    p_a, p_b = np.zeros(N), np.zeros(N)
    order = sorted(range(N), key=lambda i: len(encs[i]["ids"]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    bs = int(cfg["batch_size"])
    import time
    t0 = time.time()
    for k, chunk in enumerate(batches(N, bs)):
        idx = [order[j] for j in chunk]
        ids, att, _ = make_batch(encs, idx, pad, torch, dev)
        out = forward(model, ids, att, torch, output_hidden_states=True)
        pa, pb = p_letters(out.logits[:, -1], lids, torch)
        p_a[idx], p_b[idx] = pa, pb
        st = torch.stack([h[:, -1] for h in out.hidden_states], 1)
        last[idx] = st.to(torch.float16).cpu().numpy()
        if k % 100 == 0 or chunk[-1] == N - 1:
            done = chunk[-1] + 1
            rate = done / max(time.time() - t0, 1e-6)
            log(f"  {done}/{N} prompts ({rate:.1f}/s, about {(N - done) / max(rate, 1e-6) / 60:.0f} min left)")
    last.flush()
    pn = np.clip(p_a / np.maximum(p_a + p_b, 1e-12), 1e-4, 1 - 1e-4)
    lo = np.log(pn / (1 - pn))
    np.savez(acts_path(cfg, "meta.npz"), p_a=p_a, p_b=p_b, logodds=lo, ids=np.array([r["id"] for r in rows]),
             n_layers=sh["n_layers"], width=d)
    mass = p_a + p_b
    log(f"record done. Probability on A/B: mean {mass.mean():.3f}, min {mass.min():.3f}")
    check = {"letter_mass_mean": float(mass.mean()), "letter_mass_min": float(mass.min())}
    check.update(compare_pilot4(cfg, rows, last))
    save_json(check, Path(cfg["res"]) / "record_check.json")


def compare_pilot4(cfg, rows, last) -> dict:
    """Same prompts, same model: pilot 5's last-position states should match pilot 4's (bf16, eager vs sdpa)."""
    p4 = Path(cfg.get("pilot4_acts", ""))
    if cfg.get("quick_mode") or not (p4 / "resid.npy").exists() or not (p4 / "row_ids.json").exists():
        log("  (pilot 4's recordings not found or quick mode: consistency check skipped)")
        return {"pilot4_check": "skipped"}
    ids4 = json.load(open(p4 / "row_ids.json"))["ids"]
    m4 = np.load(p4 / "meta.npz")
    t1 = [str(x) for x in m4["position_names"]].index("t-1")
    where4 = {k: i for i, k in enumerate(ids4)}
    resid4 = np.load(p4 / "resid.npy", mmap_mode="r")
    test = [r for r in rows if r["group"] == "test"][:: max(1, sum(r["group"] == "test" for r in rows) // 40)][:40]
    li = int(cfg["primary_layer"])
    a = np.stack([np.asarray(last[r["index"], li], np.float32) for r in test])
    b = np.stack([np.asarray(resid4[where4[r["id"]], li, t1], np.float32) for r in test])
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    log(f"  pilot 4 check at layer {li} (40 prompts): relative difference median {np.median(rel):.4f}, "
        f"max {rel.max():.4f}; cosine min {cos.min():.5f}")
    return {"pilot4_check": {"layer": li, "rel_diff_median": float(np.median(rel)), "rel_diff_max": float(rel.max()),
                             "cos_min": float(cos.min())}}
