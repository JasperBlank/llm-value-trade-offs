"""Stage `formats` (extension, added 2 Nov before any result was seen): is the last-position value
code a code for value, or machinery of the A/B letter format?

The prompts of formats.jsonl (numbered "1)/2)", a yes/no "should the council choose the first
project", and a single outcome rated 1-9) are run through the model. For each format:
  (a) behaviour  does the answer follow the same values? choice formats: log-odds of the first
                 answer ("1", "Yes") against V_first - V_second; rating: the expected digit
                 against V of the outcome.
  (b) readable   A's arrow (and B's), fitted on the A/B prompts, read at the last position: the
                 option-level correlation of its readout with the goods' values (9 real options;
                 the arrow never saw this format, so every good is new to it in this format).
  (c) causal     the pushes of the steering test at the last position: A's arrow at the primary
                 layer (-2, -1, +1, +2) and at the last steering layer (-2, +2), B's arrow and a
                 random direction (-2, +2). Slope of the answer per value unit; also relative to
                 the format's own behavioural slope (1 = the push acts like that much real value)
                 and, for the numbered format, relative to the A/B slope.
  (d) writers    numbered format: the same exact decomposition of A's readout at the primary
                 layer; overlap of its top 20 pieces with the A/B top 20, and the correlation of
                 all pieces' shares between the two formats.
Writes results/<run>/formats.json and fig_formats.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import (ROOT, Hooks, acts_path, batches, bootstrap_mean_ci, forward, layer_output, log, make_batch,
                    model_shape, save_json, values)
from steer import push_last

ANSWERS = {"numbered": ("1", "2"), "should": ("Yes", "No")}
DIGITS = [str(d) for d in range(1, 10)]


def token_ids(tok, word):
    ids = set()
    for v in (word, " " + word):
        e = tok.encode(v, add_special_tokens=False)
        if e:
            ids.add(e[0])
    return sorted(ids)


def load_formats(cfg):
    rows = [json.loads(l) for l in open(ROOT / "formats.jsonl", encoding="utf-8")]
    lim = cfg.get("formats_limit")
    if lim:
        rng = np.random.default_rng(0)
        keep = []
        for fm in ("numbered", "should", "rating"):
            rs = [r for r in rows if r["format"] == fm]
            keep += [rs[i] for i in sorted(rng.choice(len(rs), size=min(lim, len(rs)), replace=False))]
        rows = keep
    return rows


def option_r(read, keys, logq, u):
    """Per option: mean readout after removing its linear dependence on log(number); correlation with u
    over the real options."""
    keys = np.array(keys)
    Z = np.column_stack([np.ones(len(read)), logq])
    res = read - Z @ np.linalg.lstsq(Z, read, rcond=None)[0]
    real = [k for k in sorted(set(keys)) if not k.startswith("N_")]
    est = {k: float(res[keys == k].mean()) for k in real}
    if len(real) < 4:
        return float("nan"), est
    return float(np.corrcoef([u[k] for k in real], [est[k] for k in real])[0, 1]), est


def cov_share(X, y):
    yc = y - y.mean()
    return (X - X.mean(0)).T @ yc / (yc @ yc)


def run(ctx):
    import torch
    cfg = ctx.cfg
    tok, model, dev = ctx.model()
    sh = model_shape(model)
    H, hd = sh["H"], sh["hd"]
    b_val, u = values()
    P = int(cfg["primary_layer"])
    Llast = int(cfg["layers"][-1])
    D = dict(np.load(Path(cfg["res"]) / "directions.npz"))
    rows = load_formats(cfg)
    encs = []
    for r in rows:
        text = tok.apply_chat_template([{"role": "user", "content": r["user"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        encs.append({"ids": tok(text, add_special_tokens=False)["input_ids"], "regions": np.zeros(1, np.int16)})
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ans = {fm: (token_ids(tok, a), token_ids(tok, b)) for fm, (a, b) in ANSWERS.items()}
    dig = [token_ids(tok, d)[0] for d in DIGITS]
    bs = int(cfg["batch_size"])
    log(f"formats: {len(rows)} prompts; answer tokens {ans}, digits {dig}")

    def measure(fm, logits):
        p = torch.softmax(logits.float(), -1)
        if fm == "rating":
            pd = p[:, dig]
            return ((pd * torch.arange(1, 10, device=pd.device)).sum(-1) / pd.sum(-1)).cpu().numpy(), pd.sum(-1).cpu().numpy()
        a, b_ = p[:, ans[fm][0]].sum(-1), p[:, ans[fm][1]].sum(-1)
        pn = (a / (a + b_)).clamp(1e-4, 1 - 1e-4)
        return torch.log(pn / (1 - pn)).cpu().numpy(), (a + b_).cpu().numpy()

    def run_set(idx, fm, state=None, push=None, keep_layers=False):
        out, mass = np.zeros(len(idx)), np.zeros(len(idx))
        reads = {L: np.zeros((len(idx), sh["d"]), np.float32) for L in cfg["layers"]} if keep_layers else None
        order = sorted(range(len(idx)), key=lambda j: len(encs[idx[j]]["ids"]))
        for chunk in batches(len(idx), bs):
            jj = [order[j] for j in chunk]
            ids, att, _ = make_batch(encs, [idx[j] for j in jj], pad, torch, dev)
            cap = {}
            with Hooks() as hk:
                if push is not None:
                    hk.fwd(*push_last(model, push[0], push[1], state))
                if keep_layers:
                    for L in cfg["layers"]:
                        hk.fwd(model.model.layers[L - 1],
                               (lambda L: lambda m, i, o: cap.__setitem__(L, layer_output(o)[:, -1].float().cpu().numpy()))(L))
                o = forward(model, ids, att, torch)
            out[jj], mass[jj] = measure(fm, o.logits[:, -1])
            if keep_layers:
                for L in cfg["layers"]:
                    reads[L][jj] = cap[L]
        return out, mass, reads

    res = {"formats": {}, "primary_layer": P}
    rng = np.random.default_rng(int(cfg["steering"]["seed"]))
    ab = json.load(open(Path(cfg["res"]) / "steering.json")) if (Path(cfg["res"]) / "steering.json").exists() else None
    for fm in ("numbered", "should", "rating"):
        idx = [i for i, r in enumerate(rows) if r["format"] == fm]
        R = [rows[i] for i in idx]
        base, mass, reads = run_set(idx, fm, keep_layers=True)
        vf = np.array([r["v_first"] for r in R])
        kf = [r["key_first"] for r in R]
        lqf = np.log([r["q_first"] for r in R])
        realf = np.array([not k.startswith("N_") for k in kf])
        rec = {"n": len(idx), "answer_mass_mean": float(mass.mean())}
        # (a) behaviour
        if fm == "rating":
            x = vf
            rec["behaviour"] = {"corr_with_value_real": float(np.corrcoef(base[realf], vf[realf])[0, 1]),
                                "corr_with_value_all": float(np.corrcoef(base, vf)[0, 1]),
                                "slope_per_value_unit_real": float(np.polyfit(vf[realf], base[realf], 1)[0]),
                                "mean_rating": float(base.mean())}
            beh_slope = rec["behaviour"]["slope_per_value_unit_real"]
            rec["behaviour"]["option_r"], _ = option_r(base, kf, lqf, u)
        else:
            dv = vf - np.array([r["v_second"] for r in R])
            rec["behaviour"] = {"corr_with_value_difference": float(np.corrcoef(base, dv)[0, 1]),
                                "slope_per_value_unit": float(np.polyfit(dv, base, 1)[0]),
                                "mean_logodds_first": float(base.mean())}
            beh_slope = rec["behaviour"]["slope_per_value_unit"]
        # (b) readable
        rec["readable"] = {}
        for L in cfg["layers"]:
            ra = (reads[L] - D[f"xma_{L}"]) @ D[f"wa_{L}"] + float(D[f"yma_{L}"])
            r_opt, est = option_r(ra, kf, lqf, u)
            entry = {"a_option_r": r_opt, "a_corr_with_value_real": float(np.corrcoef(ra[realf], vf[realf])[0, 1]),
                     "a_scale": float(np.polyfit(vf[realf], ra[realf], 1)[0]), "a_option_estimates": est}
            if fm != "rating":
                rb = (reads[L] - D[f"xmb_{L}"]) @ D[f"wb_{L}"] + float(D[f"ymb_{L}"])
                ks = [r["key_second"] for r in R]
                entry["b_option_r"], _ = option_r(rb, ks, np.log([r["q_second"] for r in R]), u)
            rec["readable"][str(L)] = entry
        # (c) causal
        if fm == "rating":
            sel = list(range(len(idx)))
        else:
            pn = 1 / (1 + np.exp(-base))
            sel = [j for j in range(len(idx)) if 0.2 < pn[j] < 0.8]
            if len(sel) > 300:
                sel = sorted(rng.choice(sel, size=300, replace=False).tolist())
            if len(sel) < 60:
                sel = sorted(sorted(range(len(idx)), key=lambda j: abs(pn[j] - 0.5))[:100])
        sidx = [idx[j] for j in sel]
        b0 = base[sel]
        conds = [("a", P, s) for s in (-2.0, -1.0, 1.0, 2.0)] + [("a", Llast, s) for s in (-2.0, 2.0)] + \
                [("random", P, s) for s in (-2.0, 2.0)] + ([("b", P, s) for s in (-2.0, 2.0)] if fm != "rating" else [])
        shifts = {}
        for name, L, s in conds:
            vec = torch.tensor(D[f"steer_{name}_{L}"], dtype=torch.float32, device=dev)
            out, _, _ = run_set(sidx, fm, state={"scale": s}, push=(L, vec))
            shifts[(name, L, s)] = out - b0
        rec["causal"] = {"n_prompts": len(sel)}
        for name, L in sorted({(c[0], c[1]) for c in conds}):
            ss = np.array(sorted(s for (n_, l_, s) in shifts if n_ == name and l_ == L))
            Y = np.stack([shifts[(name, L, s)] for s in ss], 1)
            sl = (Y * ss).sum(1) / (ss ** 2).sum()
            m, lo, hi = bootstrap_mean_ci(sl, reps=int(cfg["steering"]["bootstrap"]), seed=1)
            e = {"slope": m, "ci95": [lo, hi], "relative_to_behaviour": m / beh_slope if beh_slope else None}
            if ab and fm == "numbered" and name in ("a", "b") and str(L) in ab["per_layer"]:
                abs_ = ab["per_layer"][str(L)][name]["slope"]
                e["relative_to_ab_slope"] = m / abs_ if abs_ else None
            rec["causal"][f"{name}@{L}"] = e
        res["formats"][fm] = rec
        c = rec["causal"][f"a@{P}"]
        log(f"  {fm}: done ({len(idx)} prompts, {len(sel)} pushed)")
        del c

    # (d) writers in the numbered format
    di_name = f"v_a@{P}"
    idx = [i for i, r in enumerate(rows) if r["format"] == "numbered"]
    R = [rows[i] for i in idx]
    wa = torch.tensor(D[f"wa_{P}"], dtype=torch.float32, device=dev)
    M = [(blk.self_attn.o_proj.weight.detach().float().T @ wa).view(H, hd) for blk in model.model.layers[:P]]
    st = {}
    pieces = np.zeros((len(idx), P, H + 1), np.float32)
    emb = np.zeros(len(idx), np.float32)
    order = sorted(range(len(idx)), key=lambda j: len(encs[idx[j]]["ids"]))
    with Hooks() as hk:
        hk.fwd(model.model.embed_tokens, lambda m, i, o: st.__setitem__("emb", o[:, -1].float() @ wa))
        for li in range(P):
            blk = model.model.layers[li]
            hk.pre(blk.self_attn.o_proj, (lambda li: lambda m, inp: st.__setitem__(
                ("h", li), (inp[0][:, -1].float().view(-1, H, hd) * M[li]).sum(-1)))(li))
            hk.fwd(blk.mlp, (lambda li: lambda m, i, o: st.__setitem__(("m", li), o[:, -1].float() @ wa))(li))
        for chunk in batches(len(idx), bs):
            jj = [order[j] for j in chunk]
            ids, att, _ = make_batch(encs, [idx[j] for j in jj], pad, torch, dev)
            forward(model, ids, att, torch)
            emb[jj] = st["emb"].cpu().numpy()
            for li in range(P):
                pieces[jj, li, :H] = st[("h", li)].cpu().numpy()
                pieces[jj, li, H] = st[("m", li)].cpu().numpy()
    real = np.array([not r["key_first"].startswith("N_") for r in R])
    y = np.array([r["v_first"] for r in R])[real]
    X = np.concatenate([emb[real, None], pieces[real].reshape(int(real.sum()), -1)], 1)
    names = ["embedding"] + [f"MLP {blk}" if h == H else f"head {blk}.{h}" for blk in range(P) for h in range(H + 1)]
    share = cov_share(X, y)
    tot = float(share.sum())
    top = [names[i] for i in np.argsort(-share)[:20]]
    wr = {"total_slope": tot, "top20": top, "mlps_share": float(share[1:].reshape(P, H + 1)[:, H].sum() / tot)}
    cj = Path(cfg["res"]) / "circuit.json"
    if cj.exists():
        ab_top = [t["piece"] for t in json.load(open(cj))["writers"][di_name]["top"][:20]]
        wr["ab_top20"] = ab_top
        wr["overlap_top20"] = len(set(top) & set(ab_top))
    try:                                                                   # all pieces' shares in the A/B format
        cm = json.load(open(acts_path(cfg, "circuit_meta.json")))
        comp = np.load(acts_path(cfg, "comp.npy"), mmap_mode="r")
        embab = np.load(acts_path(cfg, "emb.npy"))
        fin = np.load(acts_path(cfg, "final.npz"))
        di = cm["directions"].index(di_name)
        test = [ctx.rows[i] for i in fin["test_index"]]
        ra_ = np.array([not r["key_a"].startswith("N_") for r in test])
        Xab = np.concatenate([embab[ra_, di:di + 1], np.asarray(comp[:, :P, :, di], np.float32)[ra_].reshape(int(ra_.sum()), -1)], 1)
        sab = cov_share(Xab, np.array([r["v_a"] for r in test])[ra_])
        wr["corr_all_shares_with_ab"] = float(np.corrcoef(share, sab)[0, 1])
    except Exception as e:                                                 # noqa: BLE001
        wr["corr_all_shares_with_ab"] = None
        log(f"  (A/B shares not available: {e})")
    res["writers_numbered"] = wr
    save_json(res, Path(cfg["res"]) / "formats.json")
    log(f"formats done (writers in the numbered format: overlap with A/B top 20 = {wr.get('overlap_top20')})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, fm in zip(axes, ("numbered", "should", "rating")):
        c = res["formats"][fm]["causal"]
        ks = [k for k in c if k != "n_prompts"]
        ax.bar(range(len(ks)), [c[k]["slope"] for k in ks], yerr=[[c[k]["slope"] - c[k]["ci95"][0] for k in ks],
                                                                  [c[k]["ci95"][1] - c[k]["slope"] for k in ks]])
        ax.set_xticks(range(len(ks)), ks, rotation=30, fontsize=8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{fm}: answer shift per value unit pushed", fontsize=9)
    fig.savefig(Path(cfg["res"]) / "fig_formats.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
