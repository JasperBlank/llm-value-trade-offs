"""Checks of the machinery on a small model (run after `pipeline.py` on the same config):

1. The steering push at layer L moves the probe's readout (read at the input of block L) by exactly
   the strength.
2. Patching every head and every MLP of every block at the last position with the donor's gives
   the donor's answer (the embedding of the last token is the same template token in both).
3. A prompt's answer is the same alone and in a padded batch.
4. The model's last hidden state (hidden_states[-1]) is after the final norm (pilot 4's note).

  python tests/test_mechanics.py --config <config used for the pipeline run>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (Hooks, encode_rows, forward, layer_output, letter_ids, load_config, load_model,  # noqa: E402
                    load_prompts, logodds, make_batch, model_shape, parse_args, replace_layer_output)
from steer import push_last  # noqa: E402


def main():
    import torch
    args = parse_args(__doc__)
    cfg = load_config(args)
    rows = load_prompts(cfg)
    tok, model, dev = load_model(cfg)
    encs = encode_rows(tok, rows)
    lids = letter_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    D = np.load(Path(cfg["res"]) / "directions.npz")
    ok = True

    # 1. the push
    L = int(cfg["primary_layer"])
    fresh = [r["index"] for r in rows if r["group"] == "fresh"][:6]
    ids, att, _ = make_batch(encs, fresh, pad, torch, dev)
    w, xm, ym = (torch.tensor(D[k], dtype=torch.float32, device=dev) for k in (f"wa_{L}", f"xma_{L}", f"yma_{L}"))
    wb_, xmb_, ymb_ = (torch.tensor(D[k], dtype=torch.float32, device=dev) for k in (f"wb_{L}", f"xmb_{L}", f"ymb_{L}"))
    vec = torch.tensor(D[f"steer_a_{L}"], dtype=torch.float32, device=dev)
    reads = {}
    for s in (0.0, 2.0, -1.0):
        cap = {}
        with Hooks() as hk:
            hk.fwd(*push_last(model, L, vec, {"scale": s}))
            hk.pre(model.model.layers[L], lambda m, inp: cap.__setitem__("x", inp[0][:, -1].float()))
            forward(model, ids, att, torch)
        reads[s] = ((cap["x"] - xm) @ w + ym).cpu().numpy()
        reads[("b", s)] = ((cap["x"] - xmb_) @ wb_ + ymb_).cpu().numpy()
    d2, d1 = reads[2.0] - reads[0.0], reads[-1.0] - reads[0.0]
    db = reads[("b", 2.0)] - reads[("b", 0.0)]
    tol = 1e-3 if cfg["dtype"] == "float32" else 0.1          # bf16 rounds the residual stream
    good = np.allclose(d2, 2.0, atol=2 * tol) and np.allclose(d1, -1.0, atol=tol) and np.allclose(db, 0, atol=2 * tol)
    ok &= good
    print(f"1. A's push moves A's readout by the strength and leaves B's: {'OK' if good else 'FAIL'} "
          f"(+2 -> {d2.round(4)}, -1 -> {d1.round(4)}; B's readout at +2: {db.round(4)})")

    # 2. patching everything at the last position gives the donor's answer
    by_id = {r["id"]: r["index"] for r in rows}
    pairs = [(by_id[r["recipient"]], r["index"]) for r in rows if r["group"] == "donor"][:6]
    rec, don = [p[0] for p in pairs], [p[1] for p in pairs]
    sh = model_shape(model)
    cache = {"z": {}, "mlp": {}}
    layers = model.model.layers
    with Hooks() as hk:
        for li, blk in enumerate(layers):
            hk.pre(blk.self_attn.o_proj, (lambda li: lambda m, inp: cache["z"].__setitem__(li, inp[0][:, -1].clone()))(li))
            hk.fwd(blk.mlp, (lambda li: lambda m, i, o: cache["mlp"].__setitem__(li, o[:, -1].clone()))(li))
        ids_d, att_d, _ = make_batch(encs, don, pad, torch, dev)
        lo_d = logodds(forward(model, ids_d, att_d, torch).logits[:, -1], lids, torch)
    ids_r, att_r, _ = make_batch(encs, rec, pad, torch, dev)
    with Hooks() as hk:
        for li, blk in enumerate(layers):
            def h_o(m, inp, li=li):
                z = inp[0].clone()
                z[:, -1] = cache["z"][li]
                return (z,) + tuple(inp[1:])

            def h_m(m, i, o, li=li):
                oo = o.clone()
                oo[:, -1] = cache["mlp"][li]
                return oo
            hk.pre(blk.self_attn.o_proj, h_o)
            hk.fwd(blk.mlp, h_m)
        lo_p = logodds(forward(model, ids_r, att_r, torch).logits[:, -1], lids, torch)
    lo_r = logodds(forward(model, ids_r, att_r, torch).logits[:, -1], lids, torch)
    good = np.allclose(lo_p, lo_d, atol=1e-3) and not np.allclose(lo_r, lo_d, atol=1e-3)
    ok &= good
    print(f"2. patching all pieces gives the donor's answer: {'OK' if good else 'FAIL'} "
          f"(recipient {lo_r.round(3)}, donor {lo_d.round(3)}, patched {lo_p.round(3)})")

    # 3. padding
    idx = [r["index"] for r in rows if r["group"] == "test"][:8]
    ids, att, _ = make_batch(encs, idx, pad, torch, dev)
    lo_b = logodds(forward(model, ids, att, torch).logits[:, -1], lids, torch)
    lo_1 = np.array([logodds(forward(model, *make_batch(encs, [i], pad, torch, dev)[:2], torch).logits[:, -1], lids, torch)[0]
                     for i in idx])
    good = np.allclose(lo_b, lo_1, atol=1e-3 if cfg["dtype"] == "float32" else 0.15)
    if cfg["dtype"] == "float32":
        ok &= good
    print(f"3. padded batch = one at a time: {'OK' if good else 'differs'} (max diff {np.abs(lo_b - lo_1).max():.2e}; "
          f"reported, not a stop in bf16)")

    # 4. the last hidden state
    cap = {}
    with Hooks() as hk:
        hk.fwd(layers[-1], lambda m, i, o: cap.__setitem__("x", layer_output(o)[:, -1].float()))
        out = forward(model, ids, att, torch, output_hidden_states=True)
    last = out.hidden_states[-1][:, -1].float()
    normed = model.model.norm(cap["x"].to(model.dtype)).float()
    print(f"4. hidden_states[-1] is after the final norm: {bool(torch.allclose(last, normed, atol=1e-4))} "
          f"(before the norm: {bool(torch.allclose(last, cap['x'], atol=1e-4))})")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
