"""Stage `directions`: the value directions at the last position, and the control directions.

For each layer in config `layers`, on pilot 4's 6,912 test prompts (pilot 3's six wordings):
  A's value direction   ridge probe for V_A (pilot 4's penalty), w_A
  B's value direction   ridge probe for V_B, w_B
Steering vectors: the shortest push that raises A's readout by exactly 1 while leaving B's readout
unchanged (and the same for B). A's and B's values move together across these prompts (the grid
is built around each pair's balance point), so w_A and w_B share a component; a push along w_A
alone would also raise B's readout.
Controls (same length as A's steering vector):
  random       a seeded random direction
  shuffled 1, 2  probes for the goods' values SHUFFLED among the 9 real options (|corr| with the true
               values < 0.1), number of people left out, fitted on prompts whose A is a real good:
               directions among the goods' representations that do not track value.
Also reported (not steering): the option-level check of pilot 4's result (leave one option out,
correlation of the 9 estimates with the true values), cos(w_A, w_B), lengths.
Writes results/<run>/directions.npz and directions.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import acts_path, log, save_json, values


def ridge(torch, X, y, c):
    """Pilot 4's ridge: penalty c times the mean eigenvalue of X'X (centred)."""
    xm, ym = X.mean(0), y.mean()
    Xc = X - xm
    n, d = Xc.shape
    if n >= d:
        G = Xc.T @ Xc
        w = torch.linalg.solve(G + c * G.diagonal().mean() * torch.eye(d, device=X.device), Xc.T @ (y - ym))
    else:
        K = Xc @ Xc.T
        w = Xc.T @ torch.linalg.solve(K + c * K.diagonal().mean() * torch.eye(n, device=X.device), y - ym)
    return w, xm, ym


def option_level_check(torch, X, y, keys, other, logq, c, u):
    """Pilot 4's test: leave one option out (train on prompts without it in either slot), per-option
    estimate with the number of people removed, correlated with u over the real options."""
    keys, other = np.array(keys), np.array(other)
    pred = np.full(len(y), np.nan)
    for k in sorted(set(keys)):
        te = keys == k
        tr = (keys != k) & (other != k)
        w, xm, ym = ridge(torch, X[torch.tensor(tr, device=X.device)], y[torch.tensor(tr, device=X.device)], c)
        pred[te] = ((X[torch.tensor(te, device=X.device)] - xm) @ w).cpu().numpy()        # relative to the fold's mean
    Z = np.column_stack([np.ones(len(pred)), logq])
    res = pred - Z @ np.linalg.lstsq(Z, pred, rcond=None)[0]
    real = [k for k in sorted(set(keys)) if not k.startswith("N_")]
    est = {k: float(res[keys == k].mean()) for k in real}
    r = float(np.corrcoef([u[k] for k in real], [est[k] for k in real])[0, 1])
    return r, est


def shuffles(u, real, n, max_abs, seed):
    rng = np.random.default_rng(seed)
    true = np.array([u[k] for k in real])
    out, tries = [], 0
    while len(out) < n and tries < 100000:
        tries += 1
        perm = rng.permutation(len(real))
        if (perm == np.arange(len(real))).any():
            continue
        cand = true[perm]
        if abs(np.corrcoef(cand, true)[0, 1]) < max_abs and not any(np.allclose(cand, o) for o in out):
            out.append(cand)
    return [{k: float(v) for k, v in zip(real, s)} for s in out]


def run(ctx):
    import torch
    cfg, rows = ctx.cfg, ctx.rows
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = cfg["steering"]
    b, u = values()
    last = np.load(acts_path(cfg, "last.npy"), mmap_mode="r")
    test = [r for r in rows if r["group"] == "test"]
    ti = np.array([r["index"] for r in test])
    ka, kb = [r["key_a"] for r in test], [r["key_b"] for r in test]
    ya = torch.tensor([r["v_a"] for r in test], dtype=torch.float32, device=dev)
    yb = torch.tensor([r["v_b"] for r in test], dtype=torch.float32, device=dev)
    lqa, lqb = np.log([r["q_a"] for r in test]), np.log([r["q_b"] for r in test])
    real = sorted({k for k in ka + kb if not k.startswith("N_")})
    shuf = shuffles(u, real, int(S["shuffles"]), float(S["shuffle_max_abs_corr"]), int(S["seed"]))
    real_a = np.array([k in real for k in ka])
    out, arrays = {"layers": cfg["layers"], "primary_layer": cfg["primary_layer"], "n_prompts": int(len(ti)),
                   "shuffled_values": shuf, "per_layer": {}}, {}
    g = torch.Generator(device="cpu").manual_seed(int(S["seed"]))
    for L in cfg["layers"]:
        X = torch.tensor(np.asarray(last[ti, L], dtype=np.float32), device=dev)
        ca, cb = float(cfg["penalty"]["v_a"][L]), float(cfg["penalty"]["v_b"][L])
        wa, xma, yma = ridge(torch, X, ya, ca)
        wb, xmb, ymb = ridge(torch, X, yb, cb)
        # Steering vectors: the shortest push that raises one readout by 1 and leaves the other unchanged.
        Wm = torch.stack([wa, wb])                                            # (2, d)
        Pm = Wm.T @ torch.linalg.inv(Wm @ Wm.T)                               # (d, 2)
        sa, sb = Pm[:, 0].contiguous(), Pm[:, 1].contiguous()
        rnd = torch.randn(sa.shape, generator=g).to(dev)
        rnd = rnd / rnd.norm() * sa.norm()
        rec = {"penalty_a": ca, "penalty_b": cb, "cos_wa_wb": float(torch.nn.functional.cosine_similarity(wa, wb, dim=0)),
               "check_push_a": [float(wa @ sa), float(wb @ sa)], "check_push_b": [float(wa @ sb), float(wb @ sb)],
               "len_plain_push_a": float((wa / (wa @ wa)).norm()),
               "steer_len_a": float(sa.norm()), "steer_len_b": float(sb.norm()),
               "mean_residual_len": float(X.norm(dim=1).mean())}
        arrays.update({f"wa_{L}": wa, f"xma_{L}": xma, f"yma_{L}": yma,
                       f"wb_{L}": wb, f"xmb_{L}": xmb, f"ymb_{L}": ymb,
                       f"steer_a_{L}": sa, f"steer_b_{L}": sb, f"steer_random_{L}": rnd})
        Xr = X[torch.tensor(real_a, device=dev)]
        for j, sv in enumerate(shuf):
            ys = torch.tensor([sv[k] for k, m in zip(ka, real_a) if m], dtype=torch.float32, device=dev)
            ws, _, _ = ridge(torch, Xr, ys, ca)
            vs = ws / ws.norm() * sa.norm()
            arrays[f"steer_shuffled{j + 1}_{L}"] = vs
            rec[f"cos_shuffled{j + 1}_wa"] = float(torch.nn.functional.cosine_similarity(ws, wa, dim=0))
        # pilot 4's option-level result, on these recordings
        rec["option_r_a"], rec["option_est_a"] = option_level_check(torch, X, ya, ka, kb, lqa, ca, u)
        rec["option_r_b"], rec["option_est_b"] = option_level_check(torch, X, yb, kb, ka, lqb, cb, u)
        out["per_layer"][str(L)] = rec
        log(f"  layer {L}: option-level r A {rec['option_r_a']:.2f}, B {rec['option_r_b']:.2f}; "
            f"cos(w_A, w_B) {rec['cos_wa_wb']:+.2f}; steering length {rec['steer_len_a']:.3g} "
            f"(residual {rec['mean_residual_len']:.3g})")
        del X, Xr
    np.savez(Path(cfg["res"]) / "directions.npz", **{k: (v.cpu().numpy() if hasattr(v, "cpu") else v) for k, v in arrays.items()})
    save_json(out, Path(cfg["res"]) / "directions.json")
    log("directions done")
