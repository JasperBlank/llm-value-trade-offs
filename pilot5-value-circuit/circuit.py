"""Stage `circuit` (exploratory): who writes the value directions, and who writes the answer.

One pass over pilot 4's test prompts. The residual stream at the last position after block L-1 is
exactly a sum: the token's embedding + the output of every attention head and every MLP in blocks
0..L-1 (Qwen3 has no biases here). Each piece is projected onto:
  w_A(L), w_B(L) for every layer L in config `layers`, and
  the logit direction: final-norm gain * (unembedding of "A" - unembedding of "B"); divided by the
  final norm's scale per prompt this is each piece's direct contribution to logit(A) - logit(B).
For the attention heads it also splits each head's contribution by the token it was read from
(attention weight x that token's value vector), summed over the parts of the prompt (REGIONS), for
A's and B's direction at the primary layer and the logit direction; and keeps the attention
weights per part of the prompt.
Writes to acts_dir: comp.npy (prompts, blocks, heads+1 [last = MLP], directions) float32,
emb.npy, resid_proj.npy (the probes' input projected, for the identity check), src.npy
(prompts, blocks, heads, regions, source directions) float16, att.npy (prompts, blocks, heads,
regions) float16, final.npz (final-norm scale, logit difference, log-odds).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import (REGIONS, Hooks, acts_path, batches, forward, layer_output, letter_ids, log, logodds,
                    main_letter_ids, make_batch, model_shape)


def direction_matrix(cfg, model, tok):
    """Columns: w_A(L), w_B(L) for each layer, then the logit direction."""
    D = np.load(Path(cfg["res"]) / "directions.npz")
    cols, names = [], []
    for L in cfg["layers"]:
        cols += [D[f"wa_{L}"], D[f"wb_{L}"]]
        names += [f"v_a@{L}", f"v_b@{L}"]
    ia, ib = main_letter_ids(tok)
    U = model.lm_head.weight.detach().float().cpu().numpy()
    gnorm = model.model.norm.weight.detach().float().cpu().numpy()
    cols.append(gnorm * (U[ia] - U[ib]))
    names.append("logit")
    return np.stack(cols, 1).astype(np.float32), names


def run(ctx):
    import torch
    cfg, rows = ctx.cfg, ctx.rows
    tok, model, dev = ctx.model()
    encs = ctx.encs()
    sh = model_shape(model)
    nL, H, KV, hd, rep, d = sh["n_layers"], sh["H"], sh["KV"], sh["hd"], sh["rep"], sh["d"]
    test = [r["index"] for r in rows if r["group"] == "test"]
    N = len(test)
    Wd_np, dnames = direction_matrix(cfg, model, tok)
    Dn = len(dnames)
    P = cfg["primary_layer"]
    src_names = [f"{t}@{P}" for t in cfg["circuit"]["source_directions"]] + ["logit"]
    src_cols = [dnames.index(s) for s in src_names]
    Ds, R = len(src_cols), len(REGIONS)
    Wd = torch.tensor(Wd_np, device=dev)
    M = [(blk.self_attn.o_proj.weight.detach().float().T @ Wd).view(H, hd, Dn) for blk in model.model.layers]
    kv_of = torch.arange(H, device=dev) // rep
    lids = letter_ids(tok)
    ia, ib = main_letter_ids(tok)
    eps = float(getattr(model.config, "rms_norm_eps", 1e-6))
    log(f"circuit: {N} test prompts; directions {dnames}; per-token sources for {src_names}")

    comp = np.lib.format.open_memmap(acts_path(cfg, "comp.npy"), mode="w+", dtype=np.float32, shape=(N, nL, H + 1, Dn))
    src = np.lib.format.open_memmap(acts_path(cfg, "src.npy"), mode="w+", dtype=np.float16, shape=(N, nL, H, R, Ds))
    att = np.lib.format.open_memmap(acts_path(cfg, "att.npy"), mode="w+", dtype=np.float16, shape=(N, nL, H, R))
    emb = np.zeros((N, Dn), np.float32)
    rproj = np.zeros((N, len(cfg["layers"]), Dn), np.float32)
    rms, xlog, ldiff, lo = (np.zeros(N, np.float32) for _ in range(4))
    st = {}

    def emb_hook(m, i, o):
        st["emb"] = (o[:, -1].float() @ Wd)

    def v_hook(li):
        def h(m, i, o):
            st["v"] = o                                                    # (B, T, KV*hd)
        return h

    def attn_hook(li):
        def h(m, i, o):
            w = o[1]                                                       # (B, H, T, T) attention weights
            a = w[:, :, -1, :].float()                                     # from the last position
            B_, T = a.shape[0], a.shape[2]
            v = st.pop("v").float().view(B_, T, KV, hd)[:, :, kv_of]       # (B, T, H, hd)
            vm = torch.einsum("bthe,hed->bthd", v, M[li][:, :, src_cols])  # (B, T, H, Ds)
            contrib = a.permute(0, 2, 1)[..., None] * vm                   # (B, T, H, Ds)
            oh = st["onehot"][:, :T]
            st["src"][li] = torch.einsum("bthd,btr->bhrd", contrib, oh)
            st["att"][li] = torch.einsum("bht,btr->bhr", a, oh)
        return h

    def o_pre(li):
        def h(m, inp):
            z = inp[0][:, -1].float().view(-1, H, hd)
            st["heads"][li] = torch.einsum("bhe,hed->bhd", z, M[li])
        return h

    def mlp_hook(li):
        def h(m, i, o):
            st["mlp"][li] = o[:, -1].float() @ Wd
        return h

    def blk_hook(li):
        def h(m, i, o):
            st["blk"][li] = layer_output(o)[:, -1].float() @ Wd
        return h

    def norm_pre(m, inp):
        x = inp[0][:, -1].float()
        st["rms"] = torch.sqrt((x * x).mean(-1) + eps)
        st["xlog"] = x @ Wd[:, -1]

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = sorted(range(N), key=lambda j: len(encs[test[j]]["ids"]))
    bs = int(cfg["batch_size"])
    with Hooks() as hk:
        hk.fwd(model.model.embed_tokens, emb_hook)
        for li, blk in enumerate(model.model.layers):
            hk.fwd(blk.self_attn.v_proj, v_hook(li))
            hk.fwd(blk.self_attn, attn_hook(li))
            hk.pre(blk.self_attn.o_proj, o_pre(li))
            hk.fwd(blk.mlp, mlp_hook(li))
            if li + 1 in cfg["layers"]:
                hk.fwd(blk, blk_hook(li))
        hk.pre(model.model.norm, norm_pre)
        import time
        t0 = time.time()
        for k, chunk in enumerate(batches(N, bs)):
            jj = [order[j] for j in chunk]
            ids, attm, regs = make_batch(encs, [test[j] for j in jj], pad, torch, dev)
            oh = np.zeros(regs.shape + (R,), np.float32)
            b_, t_ = np.nonzero(regs >= 0)
            oh[b_, t_, regs[b_, t_]] = 1
            st.update({"onehot": torch.tensor(oh, device=dev), "src": {}, "att": {}, "heads": {}, "mlp": {}, "blk": {}})
            o = forward(model, ids, attm, torch)
            heads = torch.stack([st["heads"][li] for li in range(nL)], 1)          # (B, nL, H, Dn)
            mlp = torch.stack([st["mlp"][li] for li in range(nL)], 1)[:, :, None]  # (B, nL, 1, Dn)
            comp[jj] = torch.cat([heads, mlp], 2).cpu().numpy()
            src[jj] = torch.stack([st["src"][li] for li in range(nL)], 1).half().cpu().numpy()
            att[jj] = torch.stack([st["att"][li] for li in range(nL)], 1).half().cpu().numpy()
            emb[jj] = st["emb"].cpu().numpy()
            rproj[jj] = torch.stack([st["blk"][L - 1] for L in cfg["layers"]], 1).cpu().numpy()
            rms[jj], xlog[jj] = st["rms"].cpu().numpy(), st["xlog"].cpu().numpy()
            lg = o.logits[:, -1].float()
            ldiff[jj] = (lg[:, ia] - lg[:, ib]).cpu().numpy()
            lo[jj] = logodds(o.logits[:, -1], lids, torch)
            if k % 100 == 0 or chunk[-1] == N - 1:
                done = chunk[-1] + 1
                log(f"  {done}/{N} prompts ({done / max(time.time() - t0, 1e-6):.1f}/s)")
    for a in (comp, src, att):
        a.flush()
    np.save(acts_path(cfg, "emb.npy"), emb)
    np.save(acts_path(cfg, "resid_proj.npy"), rproj)
    np.savez(acts_path(cfg, "final.npz"), rms=rms, xlog=xlog, logit_diff=ldiff, logodds=lo, test_index=np.array(test))
    json.dump({"directions": dnames, "source_directions": src_names, "regions": REGIONS, "layers": cfg["layers"],
               "n_blocks": nL, "heads": H}, open(acts_path(cfg, "circuit_meta.json"), "w"))
    # Identity checks: the pieces add up to what the probes read, and to the logit difference.
    errs = []
    for k, L in enumerate(cfg["layers"]):
        tot = emb + comp[:, :L].sum((1, 2))
        ref = rproj[:, k]
        errs.append(float(np.abs(tot - ref).max() / (np.abs(ref).max() + 1e-9)))
    tot_logit = (emb[:, -1] + comp[:, :, :, -1].sum((1, 2))) / rms
    lerr = float(np.abs(tot_logit - ldiff).max() / (np.abs(ldiff).max() + 1e-9))
    log(f"circuit done. Decomposition error (max, relative): probes {max(errs):.2e}, logit {lerr:.2e}")
    json.dump({"probe_identity_rel_err": errs, "logit_identity_rel_err": lerr},
              open(Path(cfg["res"]) / "circuit_identity.json", "w"))
