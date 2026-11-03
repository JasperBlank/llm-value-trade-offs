"""Pilot 15: where the price is finished, and how it is compared (DESIGN.md).

  python pilot15.py            Qwen3-14B: results/real/
  python pilot15.py --quick    Qwen3-0.6B on 6 values (24 sentences): results/quick/ (a pipeline test)

Steps: (a) the readout map: states at six positions and every 4th layer for the 96 sentences and their word deletions
as option A against one mid-priced B; a price readout trained on the intact sentences, tested on the deletions (Q1);
(b) the route: single-position swaps from a donor menu (Q2); (c) pilot 5's comparator pieces at the answer (Q3);
(d) pilot 14's Q4 with a working control (Q4); the report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in ("pilot7-value-route", "pilot5-value-circuit", "pilot9-vibe-check", "pilot8-intrinsic-values", "pilot10-full-range",
          "pilot11-vibe-direction", "pilot12-recipes", "pilot13-recipe-in-words", "pilot14-price-writers", "pilot15-price-route"):
    sys.path.insert(0, str(ROOT / p))
import pilot11 as p11  # noqa: E402
import pilot14 as p14  # noqa: E402
import probe11 as pb  # noqa: E402
import probe12 as q12  # noqa: E402
from common import batches, forward, log  # noqa: E402

B_TEXT = "each help a teenager get through a difficult school year"     # pilot 13's new sentence nearest the 96's median price
POS = ["A_mean", "A_end", "A_nl", "B_end", "B_nl", "END"]
A_SIDE = ["A_mean", "A_end", "A_nl"]
ROUTE_ROLES = ["A_end", "A_nl", "B_end", "B_nl"]
ROUTE_LAYERS = [8, 16, 20, 24, 28, 32]
Q2_LAYERS = [16, 20, 24, 28, 32]
END_LAYERS = [16, 20, 24, 28, 32, 36]
SCALES = (0.01, 0.1, 1.0, 10.0)
PAIR_MIN_GAP, N_RANDOM_COMP, N_RANDOM_Q4, FLIP = 2.0, 3, 5, 10.0
Q1_MIN, Q2_MIN, Q3_MIN, Q3_RANDOM_MAX, Q4_MIN = 0.6, 0.5, 0.5, 0.1, 0.3
N = p11.N_PEOPLE


def encode_roles(tok, user, a, b):
    """Token ids of the menu, and the positions of A's tokens, A's last token, the line break after A, B's last token,
    the line break after B, and the last token."""
    s = p11.p7.chat_text(tok, user)
    e = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    off = e["offset_mapping"]
    base = s.index(user)

    def span(marker, text):
        start = base + user.index(marker) + len(marker)
        end = start + len(text)
        assert s[start:end] == text and s[end] == "\n", (user, text)
        sp = [i for i, (x, y) in enumerate(off) if y > start and x < end and y > x]
        nl = next(i for i, (x, y) in enumerate(off) if x <= end < y)
        return sp, nl
    A, A_nl = span(f"A) {N} people ", a)
    B, B_nl = span(f"B) {N} people ", b)
    return e["input_ids"], {"A": A, "A_end": A[-1], "A_nl": A_nl, "B_end": B[-1], "B_nl": B_nl, "END": len(e["input_ids"]) - 1}


def menu(a, b):
    return p11.menu_user(a, b)


class Runner15(p14.Runner14):
    def _batch_roles(self, encs, ch):
        torch = self.torch
        T = max(len(encs[i][0]) for i in ch)
        ids = torch.full((len(ch), T), self.pad, dtype=torch.long)
        att = torch.zeros((len(ch), T), dtype=torch.long)
        roles = []
        for j, i in enumerate(ch):
            n = len(encs[i][0])
            o = T - n
            ids[j, o:] = torch.tensor(encs[i][0])
            att[j, o:] = 1
            r = encs[i][1]
            roles.append({"A": [o + k for k in r["A"]], **{k: o + r[k] for k in ("A_end", "A_nl", "B_end", "B_nl", "END")}})
        return ids.to(self.dev), att.to(self.dev), roles

    def _logit(self, out):
        torch = self.torch
        lp = torch.log_softmax(out.logits[:, -1].float(), -1)
        return (torch.logsumexp(lp[:, self.AB[0]], -1) - torch.logsumexp(lp[:, self.AB[1]], -1)).cpu().numpy()

    def read_roles(self, encs, layers, bs=16):
        """States at the six positions (A's tokens averaged) after each layer in `layers`, and logit P(A)."""
        torch = self.torch
        n = len(encs)
        X = np.zeros((n, len(POS), len(layers), self.d), dtype=np.float16)
        lg = np.zeros(n)
        for ch in batches(n, bs):
            ids, att, roles = self._batch_roles(encs, ch)
            cap = {}
            hooks = []
            for li, L in enumerate(layers):
                mod = self.model.model.embed_tokens if L == 0 else self.blocks[L - 1]
                hooks.append(mod.register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(li, o[0] if isinstance(o, tuple) else o)))
            try:
                with torch.no_grad():
                    lg[ch] = self._logit(forward(self.model, ids, att, torch))
            finally:
                for h in hooks:
                    h.remove()
            with torch.no_grad():
                for li in range(len(layers)):
                    H = cap[li]
                    for j, i in enumerate(ch):
                        r = roles[j]
                        rows = [H[j, r["A"]].float().mean(0)] + [H[j, r[k]].float() for k in POS[1:]]
                        X[i, :, li] = torch.stack(rows).cpu().numpy().astype(np.float16)
        return X, lg

    def patch(self, rec, don, conds, bs=8):
        """Pairs (recipient menu, donor menu). conds: ("resid", role, L) replaces the recipient's state at `role` after
        layer L with the donor's; ("pieces", parts) replaces those heads' and MLPs' outputs at the last token.
        Returns the recipient's and donor's logits and each condition's change in the recipient's logit."""
        torch = self.torch
        n = len(rec)
        res_layers = sorted({c[2] for c in conds if c[0] == "resid"})
        pblocks = sorted({p[1] for c in conds if c[0] == "pieces" for p in c[1]})
        base_r, base_d, obs = np.zeros(n), np.zeros(n), np.zeros((len(conds), n))
        for ch in batches(n, bs):
            idd, atd, rd = self._batch_roles(don, ch)
            cache = {"res": {}, "z": {}, "m": {}}
            hooks = [self.blocks[L - 1].register_forward_hook(
                lambda m, i, o, L=L: cache["res"].__setitem__(L, (o[0] if isinstance(o, tuple) else o).clone())) for L in res_layers]
            for b in pblocks:
                hooks.append(self.blocks[b].self_attn.o_proj.register_forward_pre_hook(lambda m, i, b=b: cache["z"].__setitem__(b, i[0][:, -1].clone())))
                hooks.append(self.blocks[b].mlp.register_forward_hook(lambda m, i, o, b=b: cache["m"].__setitem__(b, o[:, -1].clone())))
            try:
                with torch.no_grad():
                    base_d[ch] = self._logit(forward(self.model, idd, atd, torch))
            finally:
                for h in hooks:
                    h.remove()
            idr, atr, rr = self._batch_roles(rec, ch)
            with torch.no_grad():
                base_r[ch] = self._logit(forward(self.model, idr, atr, torch))
            for ci, c in enumerate(conds):
                hooks = []
                if c[0] == "resid":
                    _, role, L = c

                    def hk(m, i, o, L=L, role=role):
                        hh = (o[0] if isinstance(o, tuple) else o).clone()
                        for j in range(len(ch)):
                            hh[j, rr[j][role]] = cache["res"][L][j, rd[j][role]]
                        return (hh,) + tuple(o[1:]) if isinstance(o, tuple) else hh
                    hooks.append(self.blocks[L - 1].register_forward_hook(hk))
                else:
                    heads, mlps = {}, set()
                    for p in c[1]:
                        if p[0] == "mlp":
                            mlps.add(p[1])
                        else:
                            heads.setdefault(p[1], []).append(p[2])
                    for b, hs in heads.items():
                        def pre(m, inp, b=b, hs=hs):
                            z = inp[0].clone()
                            for h in hs:
                                z[:, -1, h * self.hd:(h + 1) * self.hd] = cache["z"][b][:, h * self.hd:(h + 1) * self.hd]
                            return (z,) + tuple(inp[1:])
                        hooks.append(self.blocks[b].self_attn.o_proj.register_forward_pre_hook(pre))
                    for b in mlps:
                        def fw(m, i, o, b=b):
                            oo = o.clone()
                            oo[:, -1] = cache["m"][b]
                            return oo
                        hooks.append(self.blocks[b].mlp.register_forward_hook(fw))
                try:
                    with torch.no_grad():
                        obs[ci, ch] = self._logit(forward(self.model, idr, atr, torch)) - base_r[ch]
                finally:
                    for h in hooks:
                        h.remove()
        return base_r, base_d, obs


def transfer(obs, eff, reps=1000, seed=0):
    """Slope through zero of the observed change on the donor's effect (pilot 5), with a bootstrap over pairs."""
    rng = np.random.default_rng(seed)
    t = float((obs * eff).sum() / (eff * eff).sum())
    bs = []
    for _ in range(reps):
        i = rng.integers(0, len(obs), len(obs))
        bs.append((obs[i] * eff[i]).sum() / (eff[i] * eff[i]).sum())
    return {"transfer": t, "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}


def t_readout(Xo, yo, Xv, groups):
    """Ridge trained on the intact sentences (standardised states, linear kernel, penalty by leaving one value out):
    held-out r on them, and predictions for the variants and (in sample) the intact ones."""
    mu, sd = Xo.mean(0), Xo.std(0) + 1e-6
    Ao, Av = (Xo - mu) / sd, (Xv - mu) / sd
    K, d = Ao @ Ao.T, Xo.shape[1]
    errs = [sum(np.sum((q12._ridge_k(K, yo, groups != g, groups == g, s * d) - yo[groups == g]) ** 2) for g in np.unique(groups))
            for s in SCALES]
    s = SCALES[int(np.argmin(errs))]
    held = q12.lovo_ridge_K(K, yo, groups, d, SCALES)
    coef = np.linalg.solve(K + s * d * np.eye(len(yo)), yo - yo.mean())
    return pb.r(held, yo), yo.mean() + Av @ Ao.T @ coef, yo.mean() + K @ coef


def u_readout(dX, dy, groups, torch, dev):
    """Upper bound: ridge trained on the deletions themselves (state change -> choice change), one value left out;
    the best of the penalties (so slightly optimistic)."""
    A = torch.tensor((dX - dX.mean(0)) / (dX.std(0) + 1e-6), dtype=torch.float32, device=dev)
    K = A @ A.T
    y = torch.tensor(dy, dtype=torch.float32, device=dev)
    d = dX.shape[1]
    best = -1.0
    for s in SCALES:
        pred = torch.zeros_like(y)
        for g in np.unique(groups):
            te = torch.tensor(groups == g, device=dev)
            tr = ~te
            ym = y[tr].mean()
            c = torch.linalg.solve(K[tr][:, tr] + s * d * torch.eye(int(tr.sum()), device=dev), (y[tr] - ym)[:, None])[:, 0]
            pred[te] = ym + K[te][:, tr] @ c
        best = max(best, pb.r(pred.cpu().numpy(), dy))
    return best


def parse_piece(name):
    """Pilot 5's piece names, blocks counted from 0: 'MLP 26' -> ('mlp', 26); 'head 27.14' -> ('head', 27, 14)."""
    kind, rest = name.split(" ")
    if kind == "MLP":
        return ("mlp", int(rest))
    b, h = rest.split(".")
    return ("head", int(b), int(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    model_name = "Qwen/Qwen3-0.6B" if a.quick else "Qwen/Qwen3-14B"
    res = HERE / "results" / ("quick" if a.quick else "real")
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S = p11.items(a.quick)
    vals = sorted({s["value"] for s in S})
    u = np.array([s["u"] for s in S])
    n = len(S)
    groups = np.array([vals.index(s["value"]) for s in S])
    M = Runner15(model_name)
    layers = list(range(0, M.n_layers + 1, 4))
    log(f"{model_name}: {n} sentences; B = {B_TEXT!r}")

    # ---- (a) the readout map
    enc_o = [encode_roles(M.tok, menu(s["text"], B_TEXT), s["text"], B_TEXT) for s in S]
    var_key, enc_v = [], []
    for i, s in enumerate(S):
        w = s["text"].split(" ")
        for k in range(len(w)):
            t = " ".join(w[:k] + w[k + 1:])
            var_key.append((i, k))
            enc_v.append(encode_roles(M.tok, menu(t, B_TEXT), t, B_TEXT))
    same_nl = float(np.mean([e[1]["A_nl"] == e[1]["A_end"] for e in enc_o]))
    Xo, yo = M.read_roles(enc_o, layers)
    Xv, yv = M.read_roles(enc_v, layers)
    parent = np.array([i for i, _ in var_key])
    dy = yv - yo[parent]
    vgroups = groups[parent]
    halfA = np.isin(vgroups, np.arange(0, len(vals), 2))
    log(f"read {n} + {len(var_key)} menus; choice r with price {pb.r(yo, u):.2f}; median |change| {np.median(np.abs(dy)):.2f} logits; "
        f"A's line break shares A's last token in {same_nl:.0%}")
    rT, rTa, rTb, heldT, rU, flips = {}, {}, {}, {}, {}, {}
    big = np.abs(dy) > FLIP
    for pi, pname in enumerate(POS):
        for li, L in enumerate(layers):
            xo = Xo[:, pi, li].astype(np.float64)
            xv = Xv[:, pi, li].astype(np.float64)
            held, pv, po = t_readout(xo, yo, xv, groups)
            dp = pv - po[parent]
            key = f"{pname}@{L}"
            rT[key], rTa[key], rTb[key], heldT[key] = pb.r(dp, dy), pb.r(dp[halfA], dy[halfA]), pb.r(dp[~halfA], dy[~halfA]), held
            flips[key] = float(np.mean(np.sign(dp[big]) == np.sign(dy[big]))) if big.any() else None
            rU[key] = u_readout(xv - xo[parent], dy, vgroups, M.torch, M.dev)
        log(f"readout {pname}: " + ", ".join(f"{L}: {rT[f'{pname}@{L}']:+.2f}/{rU[f'{pname}@{L}']:.2f}" for L in layers))
    cands = [f"{p}@{L}" for p in A_SIDE for L in layers]
    best_a = max(cands, key=lambda k: rTa[k])
    q1 = rTb[best_a]
    log(f"Q1: best A-side readout {best_a} (half A r {rTa[best_a]:.2f}); r on the other half {q1:.2f}")

    # ---- (b) the route, and (c) the comparator
    rng = np.random.default_rng(1500)
    donors = []
    for i, s in enumerate(S):
        c = [j for j, t in enumerate(S) if t["value"] != s["value"] and abs(t["u"] - s["u"]) >= PAIR_MIN_GAP]
        donors.append(int(rng.choice(c)))
    enc_d = [enc_o[j] for j in donors]
    conds = [("resid", r_, L) for r_ in ROUTE_ROLES for L in ROUTE_LAYERS if L <= M.n_layers]
    conds += [("resid", "END", L) for L in END_LAYERS if L <= M.n_layers]
    P5 = json.load(open(ROOT / "pilot5-value-circuit/results/real/patching.json"))["pieces"]["top-20"]
    top20 = [p for p in map(parse_piece, P5) if p[1] < M.n_layers and (p[0] == "mlp" or p[2] < M.H)]
    comp = {"top-20": top20, "top-20 MLPs": [p for p in top20 if p[0] == "mlp"], "top-20 heads": [p for p in top20 if p[0] == "head"]}
    rc = np.random.default_rng(1502)
    for k in range(N_RANDOM_COMP):
        used, rs = set(top20), []
        for p in top20:
            cs = [("head", p[1], h) for h in range(M.H)] + [("mlp", p[1])]
            cs = [x for x in cs if x not in used]
            x = cs[rc.integers(len(cs))]
            used.add(x)
            rs.append(x)
        comp[f"random-20 #{k + 1}"] = rs
    conds += [("pieces", v) for v in comp.values()]
    base_r, base_d, obs = M.patch(enc_o, enc_d, conds)
    eff = base_d - base_r
    route, compT = {}, {}
    for ci, c in enumerate(conds):
        t = transfer(obs[ci], eff)
        if c[0] == "resid":
            route[f"{c[1]}@{c[2]}"] = t
        else:
            compT[list(comp)[ci - (len(conds) - len(comp))]] = t
    q2_key = max((f"{r_}@{L}" for r_ in ROUTE_ROLES for L in Q2_LAYERS if f"{r_}@{L}" in route), key=lambda k: route[k]["transfer"])
    q2 = route[q2_key]["transfer"]
    q3 = compT["top-20"]["transfer"]
    q3_rand = max(compT[f"random-20 #{k + 1}"]["transfer"] for k in range(N_RANDOM_COMP))
    log(f"route: " + ", ".join(f"{k} {v['transfer']:+.2f}" for k, v in route.items()))
    log(f"Q2: best single position {q2_key} {q2:.2f}; Q3: comparator {q3:.2f} (random max {q3_rand:.2f})")

    # ---- (d) pilot 14's Q4 with a working control
    new = [d for d in json.load(open(ROOT / "pilot13-recipe-in-words/results/real/features13.json")) if d["set"] == "new"]
    P8 = p14.pick_partners(new, 8)
    V14 = json.load(open(ROOT / "pilot14-price-writers/results/real/verdicts14.json"))
    top10 = [("mlp", int(t["part"].split(" ")[1]) - 1) for t in V14["Q3"]["top"] if t["part"].startswith("MLP")]
    top10 = [p for p in top10 if p[1] < M.n_layers]
    pool = [("mlp", b) for b in list(range(1, 9)) + [10] + list(range(20, 30)) if b < M.n_layers and ("mlp", b) not in top10]
    rq = np.random.default_rng(1501)
    rand_sets = [[pool[k] for k in rq.choice(len(pool), size=min(len(top10), len(pool)), replace=False)] for _ in range(N_RANDOM_Q4)]
    n_mean = max(p[1] for p in top10 + pool) + 1
    enc_A = [p11.encode(M.tok, menu(s["text"], P8[0]["text"]), p14.MARK, s["text"]) for s in S]
    vrand = np.random.default_rng(0).normal(size=M.d)
    A14 = M.attribute(enc_A, vrand / np.linalg.norm(vrand), min(20, M.n_layers), n_mean)
    means = {"zmean": A14["zmean"], "mmean": A14["mmean"]}
    menu8 = [p11.encode(M.tok, menu(s["text"], d["text"]), p14.MARK, s["text"]) for s in S for d in P8]
    m8 = lambda abl=None: M.menu_logit_abl(menu8, abl).reshape(n, len(P8)).mean(1)
    b8 = p14.slope(m8(), u)
    lost_top = 1 - p14.slope(m8(p14.as_abl(top10, means)), u) / b8
    lost_rand = [1 - p14.slope(m8(p14.as_abl(rs, means)), u) / b8 for rs in rand_sets]
    q4 = bool(lost_top >= Q4_MIN and lost_top > max(lost_rand))
    log(f"Q4: lost {lost_top:.2f} with pilot 14's top 10; random sets " + ", ".join(f"{x:.2f}" for x in lost_rand))

    # ---- save and report
    ok = lambda b: "pass" if b else "fail"
    first80 = {}
    for p in POS:
        hit = [L for L in layers if flips[f"{p}@{L}"] is not None and flips[f"{p}@{L}"] >= 0.8]
        first80[p] = hit[0] if hit else None
    V = {"model": model_name, "n": n, "n_deletions": len(var_key), "B": B_TEXT, "layers": layers, "choice_r_with_price": pb.r(yo, u),
         "A_nl_same_token_as_A_end": same_nl, "n_big_changes": int(big.sum()),
         "Q1": {"best": best_a, "r_selection_half": rTa[best_a], "r": q1, "pass": bool(q1 >= Q1_MIN)},
         "Q2": {"best": q2_key, "transfer": q2, "ci95": route[q2_key]["ci95"], "pass": bool(q2 >= Q2_MIN)},
         "Q3": {"transfer": q3, "ci95": compT["top-20"]["ci95"], "random_max": q3_rand, "pass": bool(q3 >= Q3_MIN and q3_rand <= Q3_RANDOM_MAX)},
         "Q4": {"lost_top": float(lost_top), "lost_random": [float(x) for x in lost_rand], "pass": q4,
                "top10": [f"MLP {p[1] + 1}" for p in top10], "random_sets": [[f"MLP {p[1] + 1}" for p in rs] for rs in rand_sets]},
         "readout_T": rT, "readout_T_heldout_on_intact": heldT, "readout_U": rU, "flip_sign_agreement": flips, "first_layer_80pct_flips": first80,
         "route": route, "comparator": compT, "comparator_pieces": {k: [str(p) for p in v] for k, v in comp.items()},
         "pairs_effect_sd": float(np.std(eff)), "minutes": (time.time() - t0) / 60}
    json.dump(V, open(res / "verdicts15.json", "w"), indent=1)
    np.savez_compressed(res / "readout15.npz", yo=yo, yv=yv, dy=dy, var_key=np.array(var_key), u=u, base_r=base_r, base_d=base_d, obs=obs,
                        donors=np.array(donors))
    tab = lambda d, fmt: ["| position | " + " | ".join(str(L) for L in layers) + " |", "|---" * (len(layers) + 1) + "|"] + [
        f"| {p} | " + " | ".join(fmt(d[f'{p}@{L}']) for L in layers) + " |" for p in POS]
    rep = [f"# Pilot 15 results ({model_name})", "",
           f"{n} sentences as option A against B = \"{B_TEXT}\" (8:8), and their {len(var_key)} single-word deletions. "
           f"The choice's logit correlates {pb.r(yo, u):.2f} with pilot 10's price; the median deletion moves it {np.median(np.abs(dy)):.2f} logits.", "",
           "## Pre-registered questions", "",
           f"- **Q1** does Qwen value each option on its own before comparing? Best readout on A's own positions, chosen on half the values: {best_a} "
           f"(r {rTa[best_a]:.2f} there); on the other half r = {q1:.2f}: **{ok(q1 >= Q1_MIN)}** (bar {Q1_MIN})",
           f"- **Q2** does an option's worth funnel through one spot on its way to the answer? Best single-position swap: {q2_key}, transfer {q2:.2f} "
           f"[{route[q2_key]['ci95'][0]:.2f}, {route[q2_key]['ci95'][1]:.2f}]: **{ok(q2 >= Q2_MIN)}** (bar {Q2_MIN})",
           f"- **Q3** is the comparison the generic \"which is bigger\" machine (pilots 5-6)? Its top-20 pieces carry {q3:.2f} [{compT['top-20']['ci95'][0]:.2f}, "
           f"{compT['top-20']['ci95'][1]:.2f}]; random sets up to {q3_rand:.2f}: **{ok(V['Q3']['pass'])}** (≥ {Q3_MIN}, random ≤ {Q3_RANDOM_MAX})",
           f"- **Q4** (pilot 14's Q4 redone) silencing pilot 14's top 10 MLPs at the sentence loses {lost_top:.0%} of the choice's price slope; "
           f"5 random sets of 10 other MLPs (no block 1) lose " + ", ".join(f"{x:.0%}" for x in lost_rand) + f": **{ok(q4)}**", "",
           "## The readout map", "",
           "r between the readout's predicted change and the choice's change over all deletions. The readout was trained on the intact "
           "sentences only (T).", ""] + tab(rT, lambda x: f"{x:+.2f}") + [
           "", "Upper bound: a readout trained on the deletions themselves, one value left out (U).", ""] + tab(rU, lambda x: f"{x:.2f}") + [
           "", f"The {int(big.sum())} deletions that move the choice more than {FLIP:g} logits: the share whose sign the readout (T) gets right.", ""] + tab(
           flips, lambda x: "" if x is None else f"{x:.0%}") + [
           "", "The readout's held-out r on the intact sentences (one value left out).", ""] + tab(heldT, lambda x: f"{x:.2f}") + [
           "", "## The route", "",
           f"Transfer of a swap of A's sentence ({n} pairs, donors of another value priced ≥ {PAIR_MIN_GAP:g} apart; SD of the donors' "
           f"effect {np.std(eff):.2f} logits) when only one position's state at one layer is swapped.", "",
           "| position | " + " | ".join(str(L) for L in sorted({int(k.split('@')[1]) for k in route})) + " |",
           "|---" * (len({int(k.split('@')[1]) for k in route}) + 1) + "|"]
    for r_ in ROUTE_ROLES + ["END"]:
        Ls = sorted({int(k.split('@')[1]) for k in route})
        rep.append(f"| {r_} | " + " | ".join(f"{route[f'{r_}@{L}']['transfer']:+.2f}" if f"{r_}@{L}" in route else "" for L in Ls) + " |")
    rep += ["", "## The comparator (pilot 5's top-20 pieces, blocks counted from 0, at the answer position)", "", "| pieces | transfer (95% interval) |", "|---|---|"]
    rep += [f"| {k} | {v['transfer']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] |" for k, v in compT.items()]
    rep += ["", f"Q4 random sets: " + "; ".join(", ".join(x) for x in V["Q4"]["random_sets"]), "", f"Run time {V['minutes']:.1f} min."]
    (res / "report.md").write_text("\n".join(rep) + "\n")
    log(f"report written; {V['minutes']:.1f} min")


if __name__ == "__main__":
    main()
