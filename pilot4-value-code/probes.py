"""Pilot 4, step 2: look for the value code in the recorded residual stream (no model needed).

  python probes.py

1. Token map: on the 500 full prompts, how well each part of the prompt (mean over its tokens)
   predicts each option's value, layer by layer.
2. Value probes: for every layer and recorded position, ridge regressions to the option values,
   the value difference and the choice, cross-validated so a probe is always tested on goods
   (or pairs of goods) it never saw. Baseline: the number of people alone.
3. One currency or several: probes trained on gains tested on prevented harms (and back), and
   trained without money goods tested on money goods; cosine between probe directions.
4. The objective: goal-module prompts against their twins without the goal line: does the
   target's value readout rise (extra value) or does the decision move without it (bypass)?
Writes results/<run>/probes.json, probe_directions.npz and figures.
"""
from __future__ import annotations

import json

import numpy as np

from common import REGIONS, ROOT, acts_path, label_arrays, load_config, load_prompts, log, parse_args

TARGETS = ["v_a", "v_b", "dv", "choice", "log_qa"]


# --------------------------------------------------------------------------
# Ridge regression with cross-validation (GPU if available)
# --------------------------------------------------------------------------
class Ridge:
    def __init__(self, torch, dev, cs):
        self.t, self.dev, self.cs = torch, dev, cs

    def _solve_all(self, Xc, Yc):
        """Ridge weights for every c, sharing one Gram matrix. Yc (n, k) -> {c: W (d, k)}."""
        t = self.t
        n, d = Xc.shape
        out = {}
        if float((Xc * Xc).sum()) < 1e-10:            # no variation (e.g. the same token in every prompt)
            return {c: t.zeros((d, Yc.shape[1]), device=self.dev) for c in self.cs}
        if n >= d:
            G, XY = Xc.T @ Xc, Xc.T @ Yc
            scale = G.diagonal().mean()
            for c in self.cs:
                out[c] = t.linalg.solve(G + c * scale * t.eye(d, device=self.dev), XY)
        else:
            K = Xc @ Xc.T
            scale = K.diagonal().mean()
            for c in self.cs:
                out[c] = Xc.T @ t.linalg.solve(K + c * scale * t.eye(n, device=self.dev), Yc)
        return out

    def fit(self, X, y, c):
        """X (n, d), y (n,) -> (w, x_mean, y_mean) for one penalty c."""
        xm, ym = X.mean(0), y.mean()
        w = self._solve_all(X - xm, (y - ym)[:, None])
        return self._pick(w, c)[:, 0], xm, ym

    def _pick(self, ws, c):
        return ws[c] if c in ws else ws[min(ws, key=lambda k: abs(k - c))]

    def cv(self, X, Y, folds):
        """Y (n,) or (n, k). Out-of-fold predictions; the penalty is chosen per target by R2.
        Returns (predictions (n, k) for the best c of each target, best c per target)."""
        t = self.t
        one = Y.dim() == 1
        Y2 = Y[:, None] if one else Y
        preds = {c: t.full_like(Y2, float("nan")) for c in self.cs}
        self.last_offsets = t.full_like(Y2, float("nan"))      # the training mean each prediction started from
        for tr, te in folds:
            if tr.sum() < 3 or te.sum() == 0:
                continue
            xm, ym = X[tr].mean(0), Y2[tr].mean(0)
            self.last_offsets[te] = ym
            ws = self._solve_all(X[tr] - xm, Y2[tr] - ym)
            for c, W in ws.items():
                preds[c][te] = (X[te] - xm) @ W + ym
        best, P = [], t.full_like(Y2, float("nan"))
        for j in range(Y2.shape[1]):
            r2s = {c: r2_score(Y2[:, j], preds[c][:, j], t) for c in self.cs}
            bc = max(r2s, key=lambda c: r2s[c] if np.isfinite(r2s[c]) else -1e9)
            best.append(bc)
            P[:, j] = preds[bc][:, j]
        return (P[:, 0], best[0]) if one else (P, best)


def r2_score(y, p, t):
    ok = ~t.isnan(p)
    if ok.sum() < 3:
        return float("nan")
    yy, pp = y[ok], p[ok]
    return float(1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def key_folds(torch, dev, keys_a, keys_b, slot):
    """Leave one option out: test on prompts whose `slot` option is k; train on prompts without k."""
    out = []
    kk = keys_a if slot == "a" else keys_b
    for k in sorted(set(kk)):
        te = np.array([x == k for x in kk])
        tr = np.array([(a != k and b != k) for a, b in zip(keys_a, keys_b)])
        out.append((torch.tensor(tr, device=dev), torch.tensor(te, device=dev)))
    return out


def pair_folds(torch, dev, keys_a, keys_b, n_folds):
    pairs = sorted({tuple(sorted(p)) for p in zip(keys_a, keys_b)})
    rng = np.random.default_rng(0)
    assign = {p: i % n_folds for i, p in enumerate(rng.permutation(len(pairs)))}
    pid = np.array([assign[pairs.index(tuple(sorted(p)))] for p in zip(keys_a, keys_b)])
    return [(torch.tensor(pid != f, device=dev), torch.tensor(pid == f, device=dev)) for f in range(n_folds)]


def u_recovery(pred, log_q, keys, b, u):
    """Does the probe know the value of goods it never saw, beyond the number of people?
    (Our grids show less valuable goods with more people, so the number alone correlates with
    value; this measure removes it.)
    r_all / r_real: partial correlation, over held-out prompts, between the prediction and the
    true value, controlling for log(number). A probe that only tracks the number scores 0.
    estimates: per option, the mean of the prediction after removing its linear dependence on
    log(number), for comparison with the true u."""
    keys = np.array(keys)
    v = np.array([u[k] for k in keys]) + b * log_q

    def partial(m):
        if m.sum() < 5:
            return float("nan")
        Z = np.column_stack([np.ones(m.sum()), log_q[m]])
        rp = pred[m] - Z @ np.linalg.lstsq(Z, pred[m], rcond=None)[0]
        rv = v[m] - Z @ np.linalg.lstsq(Z, v[m], rcond=None)[0]
        return corr(rp, rv)
    ok = np.isfinite(pred)
    real = ok & np.array([not k.startswith("N_") for k in keys])
    Z = np.column_stack([np.ones(ok.sum()), log_q[ok]])
    res = np.full(len(pred), np.nan)
    res[ok] = pred[ok] - Z @ np.linalg.lstsq(Z, pred[ok], rcond=None)[0]
    est = {k: float(np.nanmean(res[keys == k])) for k in sorted(set(keys))}
    return {"r_all": partial(ok), "r_real": partial(real), "estimates": est}


def main():
    args = parse_args(__doc__)
    cfg = load_config(args)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_prompts(cfg)
    meta = np.load(acts_path(cfg, "meta.npz"))
    resid = np.load(acts_path(cfg, "resid.npy"), mmap_mode="r")
    vals = json.load(open(ROOT / "values.json"))
    b_val, u = vals["b"], vals["u"]
    N, L, P, d = resid.shape
    names = [str(x) for x in meta["position_names"]]
    lab = label_arrays(rows)
    pa = meta["p_a"] / np.maximum(meta["p_a"] + meta["p_b"], 1e-9)
    pc = np.clip(pa, 1e-4, 1 - 1e-4)
    lab["choice"] = np.log(pc / (1 - pc))
    test = np.array([r["group"] == "test" for r in rows])
    ti = np.where(test)[0]
    ka = [rows[i]["key_a"] for i in ti]
    kb = [rows[i]["key_b"] for i in ti]
    ridge = Ridge(torch, dev, cfg["probes"]["ridge_c"])
    Y = {k: torch.tensor(lab[k][ti], dtype=torch.float32, device=dev) for k in TARGETS}
    folds = {"a": key_folds(torch, dev, ka, kb, "a"), "b": key_folds(torch, dev, ka, kb, "b"),
             "pair": pair_folds(torch, dev, ka, kb, cfg["probes"]["pair_folds"])}
    out = {"layers": L, "positions": names, "n_test_prompts": int(len(ti)), "r2": {k: np.full((L, P), np.nan).tolist() for k in TARGETS},
           "number_baseline_r2": {}, "u_recovery": {}, "best_c": {}}
    r2 = {k: np.full((L, P), np.nan) for k in TARGETS}
    urec = {"v_a": np.full((L, P), np.nan), "v_a_real": np.full((L, P), np.nan),
            "v_b": np.full((L, P), np.nan), "v_b_real": np.full((L, P), np.nan)}
    best_c = {}

    # Number-only baseline (does not depend on the layer): V from log(number) alone, same folds.
    for tgt, q, slot in (("v_a", "log_qa", "a"), ("v_b", "log_qb", "b")):
        q_t = torch.tensor(lab[q][ti], dtype=torch.float32, device=dev)[:, None]
        p, _ = ridge.cv(q_t, Y[tgt], folds[slot])
        out["number_baseline_r2"][tgt] = r2_score(torch.tensor(lab[tgt][ti], dtype=torch.float32, device=dev), p, torch)
    dq = torch.tensor(lab["log_qa"][ti] - lab["log_qb"][ti], dtype=torch.float32, device=dev)[:, None]
    for tgt in ("dv", "choice"):
        p, _ = ridge.cv(dq, Y[tgt], folds["pair"])
        out["number_baseline_r2"][tgt] = r2_score(Y[tgt], p, torch)
    log(f"Number-only baseline R2: {out['number_baseline_r2']}")

    # ------------------------------------------------------------------ value probes
    dirs = {}
    for li in range(L):
        Xl = torch.tensor(np.asarray(resid[ti, li], dtype=np.float32), device=dev)     # (n, P, d)
        for pj in range(P):
            X = Xl[:, pj]
            for fold_name, tgts in (("a", ["v_a", "log_qa"]), ("b", ["v_b"]), ("pair", ["dv", "choice"])):
                Pm, cs_best = ridge.cv(X, torch.stack([Y[t_] for t_ in tgts], 1), folds[fold_name])
                offs = ridge.last_offsets
                for j, tgt in enumerate(tgts):
                    p = Pm[:, j]
                    r2[tgt][li, pj] = r2_score(Y[tgt], p, torch)
                    best_c[(tgt, li, pj)] = cs_best[j]
                    if tgt in ("v_a", "v_b"):
                        keys = ka if tgt == "v_a" else kb
                        q = lab["log_qa" if tgt == "v_a" else "log_qb"][ti]
                        rec = u_recovery((p - offs[:, j]).cpu().numpy(), q, keys, b_val, u)
                        urec[tgt][li, pj], urec[tgt + "_real"][li, pj] = rec["r_all"], rec["r_real"]
                        out["u_recovery"][f"{tgt}|{li}|{names[pj]}"] = rec["estimates"]
            # Full-data directions (for steering and the objective analysis) at the key positions.
            for tgt, pos in (("v_a", "a_end"), ("v_a", "a_num"), ("v_b", "b_end"), ("dv", "t-1"), ("choice", "t-1")):
                if names[pj] == pos:
                    w, xm, ym = ridge.fit(X, Y[tgt], best_c[(tgt, li, pj)])
                    dirs[f"{tgt}@{pos}"] = dirs.get(f"{tgt}@{pos}", []) + [(li, w.cpu().numpy(), xm.cpu().numpy(), float(ym))]
        log(f"  layer {li}: best R2 v_a {np.nanmax(r2['v_a'][li]):.2f}, v_b {np.nanmax(r2['v_b'][li]):.2f}, "
            f"dv {np.nanmax(r2['dv'][li]):.2f}, choice {np.nanmax(r2['choice'][li]):.2f}; "
            f"u recovery (real goods) at a_end {urec['v_a_real'][li, names.index('a_end')]:.2f}")
    out["r2"] = {k: v.tolist() for k, v in r2.items()}
    out["u_recovery_r"] = {k: v.tolist() for k, v in urec.items()}
    out["best_c"] = {f"{k[0]}|{k[1]}|{names[k[2]]}": v for k, v in best_c.items()}
    np.savez(f"{cfg['res']}/probe_directions.npz",
             **{f"{name}_w": np.stack([x[1] for x in v]) for name, v in dirs.items()},
             **{f"{name}_mean": np.stack([x[2] for x in v]) for name, v in dirs.items()},
             **{f"{name}_ymean": np.array([x[3] for x in v]) for name, v in dirs.items()},
             layers=np.arange(L))

    # ------------------------------------------------------------------ one currency or several
    log("Currency splits ...")
    gains = {k for k in set(ka + kb) if k.startswith("G_")}
    harms = {k for k in set(ka + kb) if k.startswith("H_") and "~" not in k}
    money = {"G_energy", "H_fine", "H_fine~gainword"}
    splits = {
        "gains_to_harms": ([(a in gains and b in gains) for a, b in zip(ka, kb)], [a in harms for a in ka]),
        "harms_to_gains": ([(a in harms and b in harms) for a, b in zip(ka, kb)], [a in gains for a in ka]),
        "nonmoney_to_money": ([(a not in money and b not in money) for a, b in zip(ka, kb)], [a in money for a in ka]),
    }
    cur = {name: {"r2": [], "u_error": []} for name in splits}  # u_error: partial r on the test kind
    cos = {"gains_vs_harms": [], "money_vs_nonmoney": []}
    pa_end = names.index("a_end")
    for li in range(L):
        X = torch.tensor(np.asarray(resid[ti, li, pa_end], dtype=np.float32), device=dev)
        ws = {}
        for name, (tr, te) in splits.items():
            tr_t, te_t = torch.tensor(tr, device=dev), torch.tensor(te, device=dev)
            if tr_t.sum() < 10 or te_t.sum() < 3:
                cur[name]["r2"].append(float("nan"))
                cur[name]["u_error"].append(float("nan"))
                continue
            c = best_c[("v_a", li, pa_end)]
            w, xm, ym = ridge.fit(X[tr_t], Y["v_a"][tr_t], c)
            ws[name] = w
            p = (X[te_t] - xm) @ w + ym
            cur[name]["r2"].append(r2_score(Y["v_a"][te_t], p, torch))
            keys_te = [k for k, m in zip(ka, te) if m]
            q = lab["log_qa"][ti][np.array(te)]
            rec = u_recovery(p.cpu().numpy(), q, keys_te, b_val, u)
            cur[name]["u_error"].append(rec["r_all"])
        # Directions learned on separate subsets, compared.
        sub = {"gains": [(a in gains and b in gains) for a, b in zip(ka, kb)],
               "harms": [(a in harms and b in harms) for a, b in zip(ka, kb)],
               "money": [(a in money or b in money) for a, b in zip(ka, kb)],
               "nonmoney": [(a not in money and b not in money) for a, b in zip(ka, kb)]}
        wsub = {}
        for nm, m in sub.items():
            m_t = torch.tensor(m, device=dev)
            if m_t.sum() >= 10:
                wsub[nm] = ridge.fit(X[m_t], Y["v_a"][m_t], best_c[("v_a", li, pa_end)])[0]
        cs = torch.nn.functional.cosine_similarity
        cos["gains_vs_harms"].append(float(cs(wsub["gains"], wsub["harms"], dim=0)) if {"gains", "harms"} <= set(wsub) else float("nan"))
        cos["money_vs_nonmoney"].append(float(cs(wsub["money"], wsub["nonmoney"], dim=0)) if {"money", "nonmoney"} <= set(wsub) else float("nan"))
    out["currency"] = {"transfer": cur, "cosine": cos, "position": "a_end", "target": "v_a"}

    # ------------------------------------------------------------------ the objective
    log("Objective: goal prompts against their twins ...")
    idx_of = {r["id"]: r["index"] for r in rows}
    goal_rows = [r for r in rows if r["group"] == "goal" and r["id"] + "|twin" in idx_of]
    obj = {}
    for version in ("goal1", "goal2"):
        G = [r for r in goal_rows if r["kind"] == version]
        if not G:
            continue
        gi = np.array([r["index"] for r in G])
        wi = np.array([idx_of[r["id"] + "|twin"] for r in G])
        tslot = np.array([0 if r["key_a"] == r["goal_good"] else 1 for r in G])      # where the target sits
        sign = np.where(tslot == 0, 1.0, -1.0)                                         # +: towards the target
        ch_g, ch_w = lab["choice"][gi] * sign, lab["choice"][wi] * sign
        res = {"n": int(len(G)), "choice_shift_toward_target": float(np.mean(ch_g - ch_w)),
               "target_value_readout_shift": [], "partner_value_readout_shift": [], "dv_readout_shift_toward_target": [],
               "objective_direction_cos_value": [], "objective_direction_cos_dv": [], "objective_direction_norm_ratio": [],
               "toward_target_direction_cos_dv": []}
        for li in range(L):
            def readout(name, rows_idx, pos):
                w = dirs[name][li][1]
                xm, ym = dirs[name][li][2], dirs[name][li][3]
                X = np.asarray(resid[rows_idx, li, names.index(pos)], dtype=np.float32)
                return (X - xm) @ w + ym
            va_g, va_w = readout("v_a@a_end", gi, "a_end"), readout("v_a@a_end", wi, "a_end")
            vb_g, vb_w = readout("v_b@b_end", gi, "b_end"), readout("v_b@b_end", wi, "b_end")
            t_shift = np.where(tslot == 0, va_g - va_w, vb_g - vb_w)
            p_shift = np.where(tslot == 0, vb_g - vb_w, va_g - va_w)
            dv_g, dv_w = readout("dv@t-1", gi, "t-1"), readout("dv@t-1", wi, "t-1")
            res["target_value_readout_shift"].append(float(np.mean(t_shift)))
            res["partner_value_readout_shift"].append(float(np.mean(p_shift)))
            res["dv_readout_shift_toward_target"].append(float(np.mean((dv_g - dv_w) * sign)))
            Xg = np.asarray(resid[gi, li, names.index("t-1")], dtype=np.float32)
            Xw = np.asarray(resid[wi, li, names.index("t-1")], dtype=np.float32)
            od = (Xg - Xw).mean(0)
            wv, wd = dirs["v_a@a_end"][li][1], dirs["dv@t-1"][li][1]
            res["objective_direction_cos_value"].append(float(od @ wv / (np.linalg.norm(od) * np.linalg.norm(wv) + 1e-12)))
            res["objective_direction_cos_dv"].append(float(od @ wd / (np.linalg.norm(od) * np.linalg.norm(wd) + 1e-12)))
            res["objective_direction_norm_ratio"].append(float(np.linalg.norm(od) / (np.linalg.norm(Xw, axis=1).mean() + 1e-12)))
            tw = ((Xg - Xw) * sign[:, None]).mean(0)              # the shift towards the target's side
            res["toward_target_direction_cos_dv"].append(float(tw @ wd / (np.linalg.norm(tw) * np.linalg.norm(wd) + 1e-12)))
        obj[version] = res
    out["objective"] = obj

    # ------------------------------------------------------------------ token map (full prompts)
    log("Token map on the full prompts ...")
    tm = {}
    full_rows = meta["full_rows"]
    if len(full_rows):
        resid_full = np.load(acts_path(cfg, "resid_full.npy"), mmap_mode="r")
        off, regs = meta["full_offsets"], meta["full_regions"]
        fk_a = [rows[i]["key_a"] for i in full_rows]
        fk_b = [rows[i]["key_b"] for i in full_rows]
        f_folds = key_folds(torch, dev, fk_a, fk_b, "a")
        f_folds_b = key_folds(torch, dev, fk_a, fk_b, "b")
        ya = torch.tensor(lab["v_a"][full_rows], dtype=torch.float32, device=dev)
        yb = torch.tensor(lab["v_b"][full_rows], dtype=torch.float32, device=dev)
        tm = {"regions": REGIONS, "v_a": np.full((L, len(REGIONS)), np.nan), "v_b": np.full((L, len(REGIONS)), np.nan)}
        for li in range(L):
            pooled = np.zeros((len(full_rows), len(REGIONS), d), np.float32)
            has = np.zeros((len(full_rows), len(REGIONS)), bool)
            for k in range(len(full_rows)):
                h = np.asarray(resid_full[off[k]:off[k + 1], li], dtype=np.float32)
                rg = regs[off[k]:off[k + 1]]
                for ri in range(len(REGIONS)):
                    m = rg == ri
                    if m.any():
                        pooled[k, ri], has[k, ri] = h[m].mean(0), True
            for ri in range(len(REGIONS)):
                if not has[:, ri].all():
                    continue
                X = torch.tensor(pooled[:, ri], device=dev)
                tm["v_a"][li, ri] = r2_score(ya, ridge.cv(X, ya, f_folds)[0], torch)
                tm["v_b"][li, ri] = r2_score(yb, ridge.cv(X, yb, f_folds_b)[0], torch)
        tm = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in tm.items()}
    out["token_map"] = tm

    json.dump(out, open(f"{cfg['res']}/probes.json", "w"), default=float)
    figures(cfg, out)
    log(f"Done: {cfg['res']}/probes.json")


def figures(cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res, names = cfg["res"], out["positions"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for ax, tgt in zip(axes, ["v_a", "v_b", "dv", "choice"]):
        m = np.array(out["r2"][tgt], float)
        im = ax.imshow(m, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
        ax.set_title({"v_a": "value of option A", "v_b": "value of option B", "dv": "value difference A − B",
                      "choice": "the model's choice (log-odds of A)"}[tgt], fontsize=10)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, fontsize=8)
        ax.set_xlabel("position")
    axes[0].set_ylabel("layer")
    fig.colorbar(im, ax=axes, shrink=0.8, label="held-out R²")
    fig.suptitle("Where can value be read out? (probes tested on goods / pairs they never saw)", x=0.02, ha="left")
    fig.savefig(f"{res}/fig_probe_r2.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, lab_ in (("v_a_real", "option A, real goods"), ("v_a", "option A, all goods"), ("v_b_real", "option B, real goods")):
        m = np.array(out["u_recovery_r"][k], float)
        pos = names.index("a_end" if k.startswith("v_a") else "b_end")
        ax.plot(m[:, pos], label=lab_)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("partial r with true value (number of people removed)")
    ax.set_title("Does the probe know the value of goods it never saw? (at the end of the option's line)", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(f"{res}/fig_unseen_goods.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    cur = out["currency"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, v in cur["transfer"].items():
        axes[0].plot(v["r2"], label=name.replace("_", " "))
    axes[0].set_ylim(-0.5, 1)
    axes[0].set_title("Transfer: value probe trained on one kind of good, tested on another", fontsize=10)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("R² on the other kind")
    axes[0].legend(fontsize=8)
    for name, v in cur["cosine"].items():
        axes[1].plot(v, label=name.replace("_", " "))
    axes[1].set_title("Cosine between value directions learned separately", fontsize=10)
    axes[1].set_xlabel("layer")
    axes[1].legend(fontsize=8)
    fig.savefig(f"{res}/fig_currency.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    if out["objective"]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for version, v in out["objective"].items():
            axes[0].plot(v["target_value_readout_shift"], label=f"{version}: target's value readout")
            axes[0].plot(v["dv_readout_shift_toward_target"], ls="--", label=f"{version}: value-difference readout (end)")
            axes[0].axhline(v["choice_shift_toward_target"], ls=":", color="k", lw=0.8)
            axes[1].plot(v["objective_direction_cos_value"], label=f"{version}: vs value direction")
            axes[1].plot(v["objective_direction_cos_dv"], ls="--", label=f"{version}: vs value-difference direction")
        axes[0].set_title("Shift caused by the goal line (dotted: actual shift in the choice, log-odds)", fontsize=10)
        axes[0].set_xlabel("layer")
        axes[0].set_ylabel("shift, in value units")
        axes[0].legend(fontsize=7)
        axes[1].set_title("Cosine of the 'objective present' direction (end) with the value directions", fontsize=10)
        axes[1].set_xlabel("layer")
        axes[1].legend(fontsize=7)
        fig.savefig(f"{res}/fig_objective.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    if out["token_map"]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, tgt in zip(axes, ["v_a", "v_b"]):
            m = np.array(out["token_map"][tgt], float)
            im = ax.imshow(m, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(out["token_map"]["regions"])))
            ax.set_xticklabels(out["token_map"]["regions"], rotation=60, fontsize=8)
            ax.set_title(f"value of option {'A' if tgt == 'v_a' else 'B'}, by part of the prompt", fontsize=10)
        axes[0].set_ylabel("layer")
        fig.colorbar(im, ax=axes, shrink=0.8, label="held-out R²")
        fig.savefig(f"{res}/fig_token_map.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
