"""Pilot 6: the subspace Qwen3-14B actually uses for option A's value, and a rival-hypothesis control.

  python pilot6.py                 all stages, Qwen3-14B
  python pilot6.py --quick         all stages on Qwen3-0.6B and a few pairs (a pipeline test)
  python pilot6.py --stages das,directions

Stages (DESIGN.md):
  control     the same patching as pilot 5 on the value task AND on two non-value comparisons
              (which pile has more apples; which object is heavier): where does each decision form,
              and do pilot 5's top-20 value writers carry it?            [P1, pre-registered]
  record      last-position states at the search layers (pilot 4's test prompts and the held-out
              prompts), the probe for A's value, and the top writers' outputs (for the illusion check)
  das         distributed alignment search: a k-dimensional subspace at the last position, trained so
              that swapping only it between a prompt and its donor moves the answer as much as
              swapping the whole state (main target), or as far as the donor's answer (comparison);
              evaluated on held-out pairs                                [P2, pre-registered]
  directions  the learned direction against the probe; does its coordinate track A's value on held-out
              prompts, and does it lie inside what the top writers write  [P4, pre-registered]
  steer       pushes along the learned direction, the probe and a random direction, each calibrated
              to one natural value unit                                  [P3, pre-registered]
  report      verdicts and report.md
The answer is measured as the logit difference logit("A") - logit("B") (not clamped).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
P5 = HERE.parent / "pilot5-value-circuit"
sys.path.insert(0, str(P5))
from common import (Hooks, batches, bootstrap_mean_ci, deep_update, forward, layer_output, letter_ids,  # noqa: E402
                    load_model, log, logodds, main_letter_ids, make_batch, model_shape, replace_layer_output,
                    save_json, values)

STAGES = ["control", "record", "das", "directions", "steer", "report"]


# --------------------------------------------------------------------------
# Settings, data, model
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--acts-dir", default=None)
    ap.add_argument("--stages", default=",".join(STAGES))
    return ap.parse_args()


def load_cfg(args):
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.quick:
        cfg = deep_update(cfg, cfg.get("quick", {}))
    for k, v in (("model", args.model_path), ("results_dir", args.results_dir), ("acts_dir", args.acts_dir)):
        if v:
            cfg[k] = v
    cfg["quick_mode"] = bool(args.quick)
    res = Path(cfg["results_dir"])
    res = res if res.is_absolute() else HERE / res
    res.mkdir(parents=True, exist_ok=True)
    Path(cfg["acts_dir"]).mkdir(parents=True, exist_ok=True)
    cfg["res"] = str(res)
    return cfg


class Ctx:
    def __init__(self, cfg):
        self.cfg = cfg
        lim = cfg.get("limit") or {}
        rng = np.random.default_rng(0)
        pairs = [json.loads(l) for l in open(HERE / "pairs.jsonl", encoding="utf-8")]
        self.pairs = {}
        for s in ("value_train", "value_test", "quantity", "weight"):
            ps = [p for p in pairs if p["set"] == s]
            if s in lim:
                ps = [ps[i] for i in sorted(rng.choice(len(ps), size=min(lim[s], len(ps)), replace=False))]
            self.pairs[s] = ps
        rows = [json.loads(l) for l in open(P5 / "prompts.jsonl", encoding="utf-8")]
        test = [r for r in rows if r["group"] == "test"]
        if "test_rows" in lim:
            test = [test[i] for i in sorted(rng.choice(len(test), size=min(lim["test_rows"], len(test)), replace=False))]
        self.test_rows = test
        self._m, self._enc = None, {}

    def model(self):
        if self._m is None:
            self._m = load_model(self.cfg)
        return self._m

    def encode(self, texts):
        """Token ids of the chat-formatted user messages (cached)."""
        tok = self.model()[0]
        out = []
        for t in texts:
            if t not in self._enc:
                s = tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
                self._enc[t] = {"ids": tok(s, add_special_tokens=False)["input_ids"], "regions": np.zeros(1, np.int16)}
            out.append(self._enc[t])
        return out


def piece_from_name(name, H):
    if name.startswith("MLP"):
        return int(name.split()[1]), H
    blk, h = name.split()[1].split(".")
    return int(blk), int(h)


def piece_name(blk, h, H):
    return f"MLP {blk}" if h == H else f"head {blk}.{h}"


def writer_sets(cfg, sh):
    """Pilot 5's top-20 writers of A's value (by the good's part) and 20 random pieces of the same blocks."""
    H, P = sh["H"], int(cfg["primary_layer"])
    rng = np.random.default_rng(int(cfg["control"]["random_seed"]))
    if cfg["quick_mode"] or cfg["control"]["top_writers_from"] == "none":   # small model: a stand-in set
        allp = [(b, h) for b in range(P) for h in range(H + 1)]
        m = min(20, len(allp) // 2)
        idx = rng.choice(len(allp), size=2 * m, replace=False)
        return [allp[i] for i in idx[:m]], [allp[i] for i in idx[m:]]
    pj = json.load(open((HERE / cfg["control"]["top_writers_from"]).resolve()))
    top = [piece_from_name(n, H) for n in pj["pieces"]["top-20"]]
    top50 = {piece_from_name(n, H) for n in pj["pieces"]["top-50"]}
    rest = [(b, h) for b in range(P) for h in range(H + 1) if (b, h) not in top50]
    rand = [rest[i] for i in rng.choice(len(rest), size=20, replace=False)]
    return top, rand


def diff_of(o, ia, ib):
    lg = o.logits[:, -1].float()
    return (lg[:, ia] - lg[:, ib]).detach().cpu().numpy()


def through_zero(obs, eff, reps=1000, seed=0):
    obs, eff = np.asarray(obs, float), np.asarray(eff, float)
    rng = np.random.default_rng(seed)
    t = float((obs * eff).sum() / (eff * eff).sum())
    bs = []
    for _ in range(reps):
        i = rng.integers(0, len(obs), len(obs))
        bs.append((obs[i] * eff[i]).sum() / (eff[i] * eff[i]).sum())
    return {"value": t, "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}


def last_hook(layer_module, fn):
    """Forward hook on a block: replace the last position's output with fn(h_last)."""
    def hook(m, i, o):
        h = layer_output(o)
        hn = h.clone()
        hn[:, -1] = fn(h[:, -1]).to(h.dtype)
        return replace_layer_output(o, hn)
    return layer_module, hook


# --------------------------------------------------------------------------
# Stage: control (P1)
# --------------------------------------------------------------------------
def stage_control(ctx):
    import torch
    cfg = ctx.cfg
    tok, model, dev = ctx.model()
    sh = model_shape(model)
    H, hd = sh["H"], sh["hd"]
    layers = model.model.layers
    ia, ib = main_letter_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    top, rand = writer_sets(cfg, sh)
    Ls = [L for L in cfg["control"]["patch_layers"] if L <= len(layers)]
    conds = [("resid", L) for L in Ls] + [("top20", None), ("random20", None)]
    sets = {"top20": top, "random20": rand}
    out = {"top20": [piece_name(b, h, H) for b, h in top], "random20": [piece_name(b, h, H) for b, h in rand], "tasks": {}}
    for task in ("value_test", "quantity", "weight"):
        P_ = ctx.pairs[task]
        er, ed = ctx.encode([p["user"] for p in P_]), ctx.encode([p["donor_user"] for p in P_])
        n = len(P_)
        y_r, y_d = np.zeros(n), np.zeros(n)
        obs = np.zeros((len(conds), n))
        for chunk in batches(n, int(cfg["batch_size"])):
            cache = {"z": {}, "mlp": {}, "blk": {}}
            with Hooks() as hk, torch.no_grad():
                for li, blk in enumerate(layers):
                    hk.pre(blk.self_attn.o_proj, (lambda li: lambda m, inp: cache["z"].__setitem__(li, inp[0][:, -1].clone()))(li))
                    hk.fwd(blk.mlp, (lambda li: lambda m, i, o: cache["mlp"].__setitem__(li, o[:, -1].clone()))(li))
                    hk.fwd(blk, (lambda li: lambda m, i, o: cache["blk"].__setitem__(li, layer_output(o)[:, -1].clone()))(li))
                ids, att, _ = make_batch(ed, chunk, pad, torch, dev)
                y_d[chunk] = diff_of(forward(model, ids, att, torch), ia, ib)
            ids, att, _ = make_batch(er, chunk, pad, torch, dev)
            y_r[chunk] = diff_of(forward(model, ids, att, torch), ia, ib)
            for ci, (kind, L) in enumerate(conds):
                with Hooks() as hk:
                    if kind == "resid":
                        hk.fwd(*last_hook(layers[L - 1], lambda h, L=L: cache["blk"][L - 1]))
                    else:
                        by_blk = {}
                        for b, h in sets[kind]:
                            by_blk.setdefault(b, []).append(h)
                        for b, hs in by_blk.items():
                            heads = [h for h in hs if h < H]
                            if heads:
                                def h_o(m, inp, b=b, heads=heads):
                                    z = inp[0].clone()
                                    for h in heads:
                                        z[:, -1, h * hd:(h + 1) * hd] = cache["z"][b][:, h * hd:(h + 1) * hd]
                                    return (z,) + tuple(inp[1:])
                                hk.pre(layers[b].self_attn.o_proj, h_o)
                            if H in hs:
                                def h_m(m, i, o, b=b):
                                    oo = o.clone()
                                    oo[:, -1] = cache["mlp"][b]
                                    return oo
                                hk.fwd(layers[b].mlp, h_m)
                    obs[ci, chunk] = diff_of(forward(model, ids, att, torch), ia, ib) - y_r[chunk]
        eff = y_d - y_r
        rec = {"n_pairs": n, "mean_abs_effect": float(np.abs(eff).mean()), "conditions": {}}
        for ci, (kind, L) in enumerate(conds):
            rec["conditions"][f"resid@{L}" if kind == "resid" else kind] = through_zero(obs[ci], eff)
        first = next((L for L in Ls if rec["conditions"][f"resid@{L}"]["value"] >= 0.5), None)
        rec["first_layer_half"] = first
        out["tasks"][task] = rec
        log(f"  control {task}: {n} pairs, mean |effect| {rec['mean_abs_effect']:.2f}; half-transfer layer {first}; "
            f"top-20 {rec['conditions']['top20']['value']:+.2f}, random-20 {rec['conditions']['random20']['value']:+.2f}")
    save_json(out, Path(cfg["res"]) / "control.json")


# --------------------------------------------------------------------------
# Stage: record (states, probe, writers' outputs)
# --------------------------------------------------------------------------
def stage_record(ctx):
    import torch
    sys.path.insert(0, str(P5))
    from directions import ridge
    cfg = ctx.cfg
    tok, model, dev = ctx.model()
    sh = model_shape(model)
    H, hd, d = sh["H"], sh["hd"], sh["d"]
    layers = model.model.layers
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    Ls = cfg["layers"]
    texts = [r["user"] for r in ctx.test_rows] + [p["user"] for p in ctx.pairs["value_test"]]
    enc = ctx.encode(texts)
    N = len(texts)
    states = np.zeros((N, len(Ls), d), np.float16)
    top, _ = writer_sets(cfg, sh)
    nW = min(int(cfg["writers_pca"]["n_prompts"]), len(ctx.test_rows))
    wout = np.zeros((nW, len(top), d), np.float16)
    WO = {b: layers[b].self_attn.o_proj.weight for b in {b for b, h in top if h < H}}
    order = sorted(range(N), key=lambda i: len(enc[i]["ids"]))
    with torch.no_grad():
        for chunk in batches(N, int(cfg["batch_size"])):
            idx = [order[j] for j in chunk]
            cap = {}
            with Hooks() as hk:
                for L in Ls:
                    hk.fwd(layers[L - 1], (lambda L: lambda m, i, o: cap.__setitem__(("s", L), layer_output(o)[:, -1].float()))(L))
                for b in sorted({b for b, h in top}):
                    hk.pre(layers[b].self_attn.o_proj, (lambda b: lambda m, inp: cap.__setitem__(("z", b), inp[0][:, -1].float()))(b))
                    hk.fwd(layers[b].mlp, (lambda b: lambda m, i, o: cap.__setitem__(("m", b), o[:, -1].float()))(b))
                ids, att, _ = make_batch(enc, idx, pad, torch, dev)
                forward(model, ids, att, torch)
            states[idx] = torch.stack([cap[("s", L)] for L in Ls], 1).half().cpu().numpy()
            wi = [j for j, i in enumerate(idx) if i < nW]
            if wi:
                vecs = []
                for b, h in top:
                    if h == H:
                        vecs.append(cap[("m", b)][wi])
                    else:
                        z = cap[("z", b)][wi][:, h * hd:(h + 1) * hd]
                        vecs.append(z @ WO[b][:, h * hd:(h + 1) * hd].float().T)
                wout[[idx[j] for j in wi]] = torch.stack(vecs, 1).half().cpu().numpy()
    np.save(Path(cfg["acts_dir"]) / "states.npy", states)
    # the probe for A's value (pilot 5's recipe), fitted on pilot 4's test prompts
    nt = len(ctx.test_rows)
    ya = torch.tensor([r["v_a"] for r in ctx.test_rows], dtype=torch.float32, device=dev)
    probe = {}
    for j, L in enumerate(Ls):
        X = torch.tensor(states[:nt, j].astype(np.float32), device=dev)
        w, xm, ym = ridge(torch, X, ya, float(cfg["probe_penalty"]))
        probe[f"w_{L}"], probe[f"xm_{L}"], probe[f"ym_{L}"] = w.cpu().numpy(), xm.cpu().numpy(), float(ym)
    # the space the top writers write into (for the illusion check): the directions along which their
    # outputs VARY across prompts (each piece's mean output removed), from the top singular vectors
    Wc = wout.astype(np.float32)
    Wc = Wc - Wc.mean(0, keepdims=True)
    X = torch.tensor(Wc.reshape(-1, d), device=dev)
    k = int(cfg["writers_pca"]["n_components"])
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    probe["writer_basis"] = Vh[:k].cpu().numpy()
    probe["writer_sv"] = S[:k].cpu().numpy()
    np.savez(Path(cfg["res"]) / "probe_and_writers.npz", **probe)
    log(f"  record: {N} prompts at layers {Ls}; probe fitted on {nt}; writer space from {nW} prompts x {len(top)} pieces")


# --------------------------------------------------------------------------
# Stage: das (P2)
# --------------------------------------------------------------------------
def stage_das(ctx):
    import torch
    cfg = ctx.cfg
    tok, model, dev = ctx.model()
    sh = model_shape(model)
    d = sh["d"]
    layers = model.model.layers
    ia, ib = main_letter_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    D = cfg["das"]
    PW = np.load(Path(cfg["res"]) / "probe_and_writers.npz")
    res = {"layers": {}}
    learned = {}
    data = {}
    for split in ("value_train", "value_test"):
        P_ = ctx.pairs[split]
        data[split] = {"er": ctx.encode([p["user"] for p in P_]), "ed": ctx.encode([p["donor_user"] for p in P_]), "n": len(P_)}
    for L in cfg["layers"]:
        # prepare: donor's state at L (last position), base, donor and whole-swap answers
        for split, dd in data.items():
            n = dd["n"]
            hdon = torch.zeros((n, d), dtype=torch.float32, device=dev)
            y_r, y_d, y_f = np.zeros(n), np.zeros(n), np.zeros(n)
            with torch.no_grad():
                for chunk in batches(n, int(cfg["batch_size"])):
                    cap = {}
                    with Hooks() as hk:
                        hk.fwd(layers[L - 1], lambda m, i, o: cap.__setitem__("h", layer_output(o)[:, -1].float()))
                        ids, att, _ = make_batch(dd["ed"], chunk, pad, torch, dev)
                        y_d[chunk] = diff_of(forward(model, ids, att, torch), ia, ib)
                    hdon[chunk] = cap["h"]
                    ids, att, _ = make_batch(dd["er"], chunk, pad, torch, dev)
                    y_r[chunk] = diff_of(forward(model, ids, att, torch), ia, ib)
                    with Hooks() as hk:
                        hk.fwd(*last_hook(layers[L - 1], lambda h, c=chunk: hdon[c]))
                        y_f[chunk] = diff_of(forward(model, ids, att, torch), ia, ib)
            dd.update({f"hdon_{L}": hdon, f"y_r_{L}": y_r, f"y_d_{L}": y_d, f"y_f_{L}": y_f})

        def swap_eval(W):
            """Held-out: swap the span of W's rows (k x d, orthonormal) from the donor."""
            dd = data["value_test"]
            y = np.zeros(dd["n"])
            Wt = torch.tensor(W, dtype=torch.float32, device=dev) if not torch.is_tensor(W) else W.detach()
            with torch.no_grad():
                for chunk in batches(dd["n"], int(cfg["batch_size"])):
                    ids, att, _ = make_batch(dd["er"], chunk, pad, torch, dev)
                    hdc = dd[f"hdon_{L}"][chunk]
                    with Hooks() as hk:
                        hk.fwd(*last_hook(layers[L - 1], lambda h: h.float() + ((hdc - h.float()) @ Wt.T) @ Wt))
                        y[chunk] = diff_of(forward(model, ids, att, torch), ia, ib)
            yr, yf, yd = dd[f"y_r_{L}"], dd[f"y_f_{L}"], dd[f"y_d_{L}"]
            return {"of_whole_swap": through_zero(y - yr, yf - yr), "of_donor": through_zero(y - yr, yd - yr)}

        lay = {"whole_swap_of_donor_test": through_zero(data["value_test"][f"y_f_{L}"] - data["value_test"][f"y_r_{L}"],
                                                          data["value_test"][f"y_d_{L}"] - data["value_test"][f"y_r_{L}"]),
               "runs": {}}
        w = PW[f"w_{L}"]
        lay["probe_swap"] = swap_eval((w / np.linalg.norm(w))[None])
        g = np.random.default_rng(int(D["seed"]))
        rv = g.standard_normal(d)
        lay["random_swap"] = swap_eval((rv / np.linalg.norm(rv))[None])
        runs = [("whole", k) for k in D["ks"]] + [("donor", k) for k in D["donor_target_ks"]]
        for target, k in runs:
            torch.manual_seed(int(D["seed"]) + k)
            lin = torch.nn.Linear(d, k, bias=False).to(dev)
            torch.nn.utils.parametrizations.orthogonal(lin)
            opt = torch.optim.Adam(lin.parameters(), lr=float(D["lr"]))
            dd = data["value_train"]
            tgt = torch.tensor(dd[f"y_f_{L}"] if target == "whole" else dd[f"y_d_{L}"], dtype=torch.float32, device=dev)
            hist = []
            t0 = time.time()
            rng = np.random.default_rng(int(D["seed"]))
            for ep in range(int(D["epochs"])):
                perm = rng.permutation(dd["n"])
                losses = []
                for s0 in range(0, dd["n"], int(D["batch_size"])):
                    chunk = perm[s0:s0 + int(D["batch_size"])].tolist()
                    ids, att, _ = make_batch(dd["er"], chunk, pad, torch, dev)
                    hdc = dd[f"hdon_{L}"][chunk]
                    W = lin.weight
                    with Hooks() as hk:
                        hk.fwd(*last_hook(layers[L - 1], lambda h: h.float() + ((hdc - h.float()) @ W.T) @ W))
                        try:
                            o = model(input_ids=ids, attention_mask=att, logits_to_keep=1)
                        except TypeError:
                            o = model(input_ids=ids, attention_mask=att)
                    lg = o.logits[:, -1].float()
                    y = lg[:, ia] - lg[:, ib]
                    loss = ((y - tgt[chunk]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.detach()))
                hist.append(float(np.mean(losses)))
            Wf = lin.weight.detach().float().cpu().numpy()
            ev = swap_eval(Wf)
            key = f"{target}_k{k}"
            lay["runs"][key] = {"train_loss_by_epoch": hist, "minutes": (time.time() - t0) / 60, **ev}
            learned[f"W_{L}_{key}"] = Wf
            log(f"  das L{L} {key}: held-out {ev['of_whole_swap']['value']:+.2f} of the whole-state swap "
                f"({ev['of_donor']['value']:+.2f} of the donor's answer); {(time.time() - t0) / 60:.1f} min")
        res["layers"][str(L)] = lay
        log(f"  das L{L}: whole-state swap carries {lay['whole_swap_of_donor_test']['value']:+.2f} of the donor's answer; "
            f"probe swap {lay['probe_swap']['of_whole_swap']['value']:+.2f}, random {lay['random_swap']['of_whole_swap']['value']:+.2f}")
    np.savez(Path(cfg["res"]) / "das_subspaces.npz", **learned)
    save_json(res, Path(cfg["res"]) / "das.json")


# --------------------------------------------------------------------------
# Stage: directions (P4) - the learned direction against the probe; illusion checks
# --------------------------------------------------------------------------
def stage_directions(ctx):
    cfg = ctx.cfg
    PW = np.load(Path(cfg["res"]) / "probe_and_writers.npz")
    S = np.load(Path(cfg["res"]) / "das_subspaces.npz")
    states = np.load(Path(cfg["acts_dir"]) / "states.npy").astype(np.float32)
    nt = len(ctx.test_rows)
    va_train = np.array([r["v_a"] for r in ctx.test_rows])
    real_train = np.array([not r["key_a"].startswith("N_") for r in ctx.test_rows])
    va_test = np.array([p["v_a"] for p in ctx.pairs["value_test"]])
    dv_test = va_test - np.array([p["v_b"] for p in ctx.pairs["value_test"]])
    basis = PW["writer_basis"]
    d = basis.shape[1]
    out = {"writer_space_dims": int(basis.shape[0]), "random_share": float(basis.shape[0] / d)}
    for j, L in enumerate(cfg["layers"]):
        w = PW[f"w_{L}"]
        wu = w / np.linalg.norm(w)
        Htr, Hte = states[:nt, j], states[nt:, j]
        for key in [k for k in S.files if k.startswith(f"W_{L}_")]:
            W = S[key]                                                     # (k, d)
            name = key[len(f"W_{L}_"):]
            e = {"probe_in_subspace": float(np.linalg.norm(W @ wu) ** 2),     # share of the probe inside the subspace
                 "writer_share": float(np.linalg.norm(basis @ W.T) ** 2 / W.shape[0])}
            if W.shape[0] == 1:
                r = W[0]                                           # its sign is arbitrary: orient it on the
                if np.polyfit(va_train[real_train], (Htr @ r)[real_train], 1)[0] < 0:   # training prompts only
                    r = -r
                ctr, cte = Htr @ r, Hte @ r
                e.update({"cos_with_probe": float(r @ wu),
                          "corr_with_va_train_real": float(np.corrcoef(ctr[real_train], va_train[real_train])[0, 1]),
                          "corr_with_va_heldout": float(np.corrcoef(cte, va_test)[0, 1]),
                          "corr_with_dv_heldout": float(np.corrcoef(cte, dv_test)[0, 1]),   # with V_A - V_B
                          "calibration_slope": float(np.polyfit(va_train[real_train], ctr[real_train], 1)[0])})
            out[f"{L}|{name}"] = e
            log(f"  L{L} {name}: " + ", ".join(f"{k} {v:+.3f}" for k, v in e.items()))
        cp = Hte @ wu
        out[f"{L}|probe"] = {"corr_with_va_heldout": float(np.corrcoef(cp, va_test)[0, 1]),
                             "corr_with_dv_heldout": float(np.corrcoef(cp, dv_test)[0, 1]),
                             "writer_share": float(np.linalg.norm(basis @ wu) ** 2)}
    save_json(out, Path(cfg["res"]) / "directions6.json")


# --------------------------------------------------------------------------
# Stage: steer (P3)
# --------------------------------------------------------------------------
def stage_steer(ctx):
    import torch
    cfg = ctx.cfg
    tok, model, dev = ctx.model()
    layers = model.model.layers
    ia, ib = main_letter_ids(tok)
    lids = letter_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    S, Sx = cfg["steering"], np.load(Path(cfg["res"]) / "das_subspaces.npz")
    PW = np.load(Path(cfg["res"]) / "probe_and_writers.npz")
    states = np.load(Path(cfg["acts_dir"]) / "states.npy").astype(np.float32)
    L = int(cfg["primary_layer"])
    j = cfg["layers"].index(L)
    nt = len(ctx.test_rows)
    real = np.array([not r["key_a"].startswith("N_") for r in ctx.test_rows])
    va = np.array([r["v_a"] for r in ctx.test_rows])

    def calibrated(e):
        e = e / np.linalg.norm(e)
        c = float(np.polyfit(va[real], states[:nt, j][real] @ e, 1)[0])     # coordinate change per natural value unit
        return e * c, c                                                     # push per value unit = e * c
    r = Sx[f"W_{L}_whole_k1"][0]
    vec_r, c_r = calibrated(r)
    if c_r < 0:
        vec_r, c_r = -vec_r, -c_r
    w = PW[f"w_{L}"]
    vec_w, c_w = calibrated(w)
    g = np.random.default_rng(int(S["seed"]))
    rv = g.standard_normal(r.shape[0])
    vec_rand = rv / np.linalg.norm(rv) * np.linalg.norm(vec_r)
    P_ = ctx.pairs["value_test"]
    enc = ctx.encode([p["user"] for p in P_])
    n = len(P_)

    def run(idx, vec=None, s=0.0):
        y, lo = np.zeros(len(idx)), np.zeros(len(idx))
        vt = torch.tensor(vec, dtype=torch.float32, device=dev) if vec is not None else None
        with torch.no_grad():
            for chunk in batches(len(idx), int(cfg["batch_size"])):
                ids, att, _ = make_batch(enc, [idx[c] for c in chunk], pad, torch, dev)
                with Hooks() as hk:
                    if vt is not None and s != 0:
                        hk.fwd(*last_hook(layers[L - 1], lambda h: h.float() + s * vt))
                    o = forward(model, ids, att, torch)
                y[chunk] = diff_of(o, ia, ib)
                lo[chunk] = logodds(o.logits[:, -1], lids, torch)
        return y, lo
    base_all, lo_all = run(list(range(n)))
    pa = 1 / (1 + np.exp(-lo_all))
    near = [i for i in range(n) if S["window"][0] < pa[i] < S["window"][1]]
    rng = np.random.default_rng(int(S["seed"]))
    if len(near) > S["max_prompts"]:
        near = sorted(rng.choice(near, size=int(S["max_prompts"]), replace=False).tolist())
    if len(near) < S["min_prompts"]:
        near = sorted(sorted(range(n), key=lambda i: abs(pa[i] - 0.5))[:int(S["min_prompts"])])
    dv = np.array([p["v_a"] - p["v_b"] for p in P_])
    out = {"layer": L, "n_prompts": len(near), "calibration": {"learned": c_r, "probe": c_w},
           "push_length_per_unit": {"learned": float(np.linalg.norm(vec_r)), "probe": float(np.linalg.norm(vec_w)),
                                    "random": float(np.linalg.norm(vec_rand))},
           "behaviour_slope_logit_diff_per_value_unit": float(np.polyfit(dv, base_all, 1)[0]), "directions": {}}
    b0, l0 = base_all[near], lo_all[near]
    for name, vec, strengths in (("learned", vec_r, S["strengths"]), ("probe", vec_w, S["strengths"]),
                                 ("random", vec_rand, [-2, 2])):
        shifts, lshifts = {}, {}
        for s in strengths:
            if s == 0:
                continue
            y, lo = run(near, vec, float(s))
            shifts[s], lshifts[s] = y - b0, lo - l0
        ss = np.array(sorted(shifts))
        sl = (np.stack([shifts[s] for s in ss], 1) * ss).sum(1) / (ss ** 2).sum()
        sll = (np.stack([lshifts[s] for s in ss], 1) * ss).sum(1) / (ss ** 2).sum()
        m, lo_, hi_ = bootstrap_mean_ci(sl, reps=int(S["bootstrap"]), seed=1)
        m2, lo2, hi2 = bootstrap_mean_ci(sll, reps=int(S["bootstrap"]), seed=1)
        out["directions"][name] = {"slope_logit_diff": m, "ci95": [lo_, hi_], "slope_logodds": m2, "ci95_logodds": [lo2, hi2],
                                   "mean_shift": {str(s): float(v.mean()) for s, v in shifts.items()}}
        log(f"  steer {name}: {m:+.3f} logit difference per value unit [{lo_:+.3f}, {hi_:+.3f}]")
    save_json(out, Path(cfg["res"]) / "steer6.json")


# --------------------------------------------------------------------------
# Stage: report (verdicts)
# --------------------------------------------------------------------------
def stage_report(ctx):
    cfg = ctx.cfg
    R = Path(cfg["res"])
    pr = cfg["prereg"]
    L = str(cfg["primary_layer"])
    lines = ["# Pilot 6 results", ""]
    V = {}
    if (R / "control.json").exists():
        C = json.load(open(R / "control.json"))
        v = C["tasks"]["value_test"]
        t20v, l50v = v["conditions"]["top20"]["value"], v["first_layer_half"]
        per = {}
        for t in ("quantity", "weight"):
            c = C["tasks"][t]
            t20, l50 = c["conditions"]["top20"]["value"], c["first_layer_half"]
            same_pieces = t20 >= pr["control_t20_ratio"] * t20v
            same_stage = l50 is not None and l50v is not None and abs(l50 - l50v) <= pr["control_layer_tolerance"]
            per[t] = {"top20": t20, "half_layer": l50, "same_pieces": same_pieces, "same_stage": same_stage}
        if all(p["same_pieces"] and p["same_stage"] for p in per.values()):
            verdict = "generic comparator: the non-value comparisons use the same stage and the same pieces"
        elif all(not p["same_pieces"] for p in per.values()):
            verdict = "value-specific pieces: the top value writers do not carry the non-value comparisons"
        else:
            verdict = "mixed"
        V["P1"] = {"verdict": verdict, "value": {"top20": t20v, "half_layer": l50v}, **per}
        lines += ["## P1. A value circuit or a generic comparator?", "",
                  "| task | pairs | half-transfer layer | top-20 value writers | random 20 |", "|---|---|---|---|---|"]
        for t in ("value_test", "quantity", "weight"):
            c = C["tasks"][t]
            f = lambda x: f"{x['value']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]"
            lines.append(f"| {t} | {c['n_pairs']} | {c['first_layer_half']} | {f(c['conditions']['top20'])} | {f(c['conditions']['random20'])} |")
        lines += ["", f"**P1: {verdict}.**", "", "Whole-state transfer by layer:", ""]
        for t in ("value_test", "quantity", "weight"):
            c = C["tasks"][t]
            lines.append(f"- {t}: " + ", ".join(f"{k.split('@')[1]}: {x['value']:.2f}" for k, x in c["conditions"].items() if k.startswith("resid@")))
    if (R / "das.json").exists():
        Dj = json.load(open(R / "das.json"))
        lay = Dj["layers"][L]
        k1 = lay["runs"]["whole_k1"]["of_whole_swap"]
        V["P2"] = {"k1_fraction": k1["value"], "ci95": k1["ci95"], "pass": k1["value"] >= pr["das_k1_min_fraction"]}
        lines += ["", "## P2. How few directions carry A's value into the decision? (held-out pairs)", "",
                  "| layer | run | of the whole-state swap | of the donor's answer |", "|---|---|---|---|"]
        for Ls, ly in Dj["layers"].items():
            lines.append(f"| {Ls} | whole-state swap itself | 1 | {ly['whole_swap_of_donor_test']['value']:+.2f} |")
            lines.append(f"| {Ls} | probe direction (k=1) | {ly['probe_swap']['of_whole_swap']['value']:+.2f} | {ly['probe_swap']['of_donor']['value']:+.2f} |")
            lines.append(f"| {Ls} | random direction (k=1) | {ly['random_swap']['of_whole_swap']['value']:+.2f} | {ly['random_swap']['of_donor']['value']:+.2f} |")
            for name, x in ly["runs"].items():
                lines.append(f"| {Ls} | learned, {name} | {x['of_whole_swap']['value']:+.2f} [{x['of_whole_swap']['ci95'][0]:+.2f}, "
                             f"{x['of_whole_swap']['ci95'][1]:+.2f}] | {x['of_donor']['value']:+.2f} |")
        lines += ["", f"**P2 (layer {L}, k = 1, whole-swap target): {k1['value']:.2f} of the whole-state swap; "
                  f"{'pass' if V['P2']['pass'] else 'fail'} (needs {pr['das_k1_min_fraction']}).**"]
    if (R / "directions6.json").exists():
        Dd = json.load(open(R / "directions6.json"))
        e = Dd.get(f"{L}|whole_k1", {})
        if e:
            ratio = e["writer_share"] / Dd["random_share"]
            V["P4"] = {"corr_heldout": e["corr_with_va_heldout"], "writer_share_ratio": ratio,
                       "pass": e["corr_with_va_heldout"] >= pr["illusion_min_corr"] and ratio >= pr["illusion_min_share_ratio"]}
            lines += ["", "## P4. Is the learned direction real (not an illusion)?", "",
                      f"Learned direction at layer {L}: its coordinate correlates {e['corr_with_va_heldout']:+.2f} with A's value on "
                      f"held-out prompts (probe: {Dd[f'{L}|probe']['corr_with_va_heldout']:+.2f}), and {e['corr_with_dv_heldout']:+.2f} with the "
                      f"difference V_A - V_B (probe: {Dd[f'{L}|probe']['corr_with_dv_heldout']:+.2f}); {e['writer_share']:.1%} of it lies in the "
                      f"top writers' {Dd['writer_space_dims']}-dimensional output space ({ratio:.1f} x a random direction; probe: "
                      f"{Dd[f'{L}|probe']['writer_share'] / Dd['random_share']:.1f} x); cosine with the probe {e['cos_with_probe']:+.2f}.",
                      "", f"**P4: {'pass' if V['P4']['pass'] else 'fail'}.**"]
    if (R / "steer6.json").exists():
        St = json.load(open(R / "steer6.json"))
        dl, dp, dr = (St["directions"][k] for k in ("learned", "probe", "random"))
        ok_ratio = dl["slope_logit_diff"] >= pr["steer_min_ratio"] * dp["slope_logit_diff"] and dl["ci95"][0] > 0
        ok_rand = abs(dr["slope_logit_diff"]) < pr["steer_random_max_ratio"] * dl["slope_logit_diff"]
        V["P3"] = {"learned": dl["slope_logit_diff"], "probe": dp["slope_logit_diff"], "random": dr["slope_logit_diff"],
                   "pass": bool(ok_ratio and ok_rand)}
        lines += ["", "## P3. Does pushing the learned direction move the choice?", "",
                  f"{St['n_prompts']} held-out prompts, layer {St['layer']}; each push calibrated to one natural value unit. "
                  f"The model's own sensitivity: {St['behaviour_slope_logit_diff_per_value_unit']:.2f} logit difference per value unit.", "",
                  "| direction | logit difference per unit (95%) | log-odds per unit | push length |", "|---|---|---|---|"]
        for k in ("learned", "probe", "random"):
            x = St["directions"][k]
            lines.append(f"| {k} | {x['slope_logit_diff']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] | {x['slope_logodds']:+.3f} | "
                         f"{St['push_length_per_unit'][k]:.2f} |")
        lines += ["", f"**P3: {'pass' if V['P3']['pass'] else 'fail'}.**"]
    (R / "report.md").write_text("\n".join(lines) + "\n")
    save_json(V, R / "verdicts6.json")
    log("report done: " + json.dumps({k: (v.get("verdict") or v.get("pass")) for k, v in V.items()}))


def main():
    args = parse_args()
    cfg = load_cfg(args)
    ctx = Ctx(cfg)
    log(f"Pilot 6, {cfg['model']}; pairs: " + ", ".join(f"{k} {len(v)}" for k, v in ctx.pairs.items()))
    fns = {"control": stage_control, "record": stage_record, "das": stage_das, "directions": stage_directions,
           "steer": stage_steer, "report": stage_report}
    for s in args.stages.split(","):
        t0 = time.time()
        log(f"===== stage {s} =====")
        fns[s.strip()](ctx)
        log(f"===== stage {s} finished in {(time.time() - t0) / 60:.1f} min =====")


if __name__ == "__main__":
    main()
