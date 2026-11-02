"""Mechanical checks for pilot 6 (run after the pipeline, on the same config).

  python tests/test_mechanics6.py --config <config used for the run>
  python tests/test_mechanics6.py --quick

1. The subspace swap with a basis of the WHOLE space reproduces the whole-state swap exactly,
   and with an empty basis changes nothing.
2. Swapping the last block's output reproduces the donor's answer exactly (control: transfer 1 at the top).
3. Every learned subspace has orthonormal rows, and training lowered the loss.
4. The verdicts file has P1-P4 and the report exists.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import pilot6 as p6  # noqa: E402
from common import Hooks, batches, make_batch, main_letter_ids, model_shape  # noqa: E402


def main():
    import torch
    args = p6.parse_args()
    args.stages = ""
    cfg = p6.load_cfg(args)
    ctx = p6.Ctx(cfg)
    R = Path(cfg["res"])
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        print(f"{'OK ' if cond else 'BAD'} {name} {info}")
        ok = ok and bool(cond)

    tok, model, dev = ctx.model()
    d = model_shape(model)["d"]
    layers = model.model.layers
    ia, ib = main_letter_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    P_ = ctx.pairs["value_test"][:16]
    er, ed = ctx.encode([p["user"] for p in P_]), ctx.encode([p["donor_user"] for p in P_])
    idx = list(range(len(P_)))
    L = int(cfg["primary_layer"])
    with torch.no_grad():
        cap = {}
        with Hooks() as hk:
            hk.fwd(layers[L - 1], lambda m, i, o: cap.__setitem__("h", p6.layer_output(o)[:, -1].float()))
            hk.fwd(layers[-1], lambda m, i, o: cap.__setitem__("top", p6.layer_output(o)[:, -1].clone()))
            ids, att, _ = make_batch(ed, idx, pad, torch, dev)
            y_d = p6.diff_of(p6.forward(model, ids, att, torch), ia, ib)
        ids, att, _ = make_batch(er, idx, pad, torch, dev)
        y_r = p6.diff_of(p6.forward(model, ids, att, torch), ia, ib)

        def run(fn, blk):
            with Hooks() as hk:
                hk.fwd(*p6.last_hook(blk, fn))
                return p6.diff_of(p6.forward(model, ids, att, torch), ia, ib)
        y_f = run(lambda h: cap["h"], layers[L - 1])
        I = torch.eye(d, device=dev)
        y_full = run(lambda h: h.float() + ((cap["h"] - h.float()) @ I.T) @ I, layers[L - 1])
        Z = torch.zeros((1, d), device=dev)
        y_none = run(lambda h: h.float() + ((cap["h"] - h.float()) @ Z.T) @ Z, layers[L - 1])
        y_top = run(lambda h: cap["top"], layers[-1])
    tol = 1e-3 + 0.02 * np.abs(y_f).max()
    check("1a. swap of the whole space = whole-state swap", np.abs(y_full - y_f).max() < tol, f"max diff {np.abs(y_full - y_f).max():.2e}")
    check("1b. swap of an empty space = no change", np.abs(y_none - y_r).max() < tol, f"max diff {np.abs(y_none - y_r).max():.2e}")
    check("2. last block's output swapped = donor's answer", np.abs(y_top - y_d).max() < tol, f"max diff {np.abs(y_top - y_d).max():.2e}")
    if (R / "das_subspaces.npz").exists():
        S = np.load(R / "das_subspaces.npz")
        worst = max(np.abs(S[k] @ S[k].T - np.eye(S[k].shape[0])).max() for k in S.files)
        check("3a. learned subspaces are orthonormal", worst < 1e-3, f"worst deviation {worst:.1e}")
        D = json.load(open(R / "das.json"))
        drops = [r["train_loss_by_epoch"][-1] <= r["train_loss_by_epoch"][0] * 1.05 for ly in D["layers"].values() for r in ly["runs"].values()]
        check("3b. training did not raise the loss", all(drops), f"{sum(drops)}/{len(drops)} runs")
        C = json.load(open(R / "control.json"))
        top = max(int(k.split("@")[1]) for k in C["tasks"]["value_test"]["conditions"] if k.startswith("resid@"))
        if top == len(layers):
            vals = [C["tasks"][t]["conditions"][f"resid@{top}"]["value"] for t in C["tasks"]]
            check("2b. control: transfer at the last block is 1 for all tasks", all(abs(v - 1) < 0.02 for v in vals), str(np.round(vals, 3)))
    if (R / "verdicts6.json").exists():
        V = json.load(open(R / "verdicts6.json"))
        check("4. verdicts P1-P4 and the report exist", set(V) >= {"P1", "P2", "P3", "P4"} and (R / "report.md").exists(), str(sorted(V)))
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
