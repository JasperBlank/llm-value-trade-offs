"""Pilot 17: do Llama and a bigger Qwen price outcomes the same way? (DESIGN.md)

  python pilot17.py --model llama8|qwen14|qwen32     results/real/<model>/
  python pilot17.py --model qwen06 --quick           Qwen3-0.6B on 6 values: results/quick/qwen06/ (a pipeline test)

Per model: (a) fixed or relative price: every A (pilot 10's sentences and pilot 14's word deletions) against 9
alternatives; common shares, order flips, cycles, the readout inside. (b) the comparator: where the decision arrives at
the answer position, the 20 pieces whose direct contribution carries a swap of A's sentence (chosen on half the values,
tested on the other half), and whether the same pieces carry pilot 6's apples and weight swaps.
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
          "pilot11-vibe-direction", "pilot12-recipes", "pilot13-recipe-in-words", "pilot14-price-writers", "pilot15-price-route",
          "pilot16-relative-or-absolute", "pilot17-three-models"):
    sys.path.insert(0, str(ROOT / p))
import pilot11 as p11  # noqa: E402
import pilot14 as p14  # noqa: E402
import pilot15 as p15  # noqa: E402
import pilot16 as p16  # noqa: E402
import probe11 as pb  # noqa: E402
from common import batches, forward, log, main_letter_ids  # noqa: E402

MODELS = {"llama8": "meta-llama/Llama-3.1-8B-Instruct", "qwen14": "Qwen/Qwen3-14B", "qwen32": "Qwen/Qwen3-32B", "qwen06": "Qwen/Qwen3-0.6B"}
FRACS = [0.4, 0.5, 0.6, 0.7, 0.8]
RESID_FRACS = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
TOP_K, N_RANDOM, N_CONTROL, GAP = 20, 3, 120, 2.0
Q1_MIN, Q2_MIN, Q3_MIN, RANDOM_MAX, Q2B_MIN = 0.9, 0.5, 0.3, 0.1, 0.4


def encode_plain(tok, user):
    ids = tok(p11.p7.chat_text(tok, user), add_special_tokens=False)["input_ids"]
    n = len(ids) - 1
    return ids, {"A": [n], "A_end": n, "A_nl": n, "B_end": n, "B_nl": n, "END": n}


class Runner17(p16.Runner16):
    def __init__(self, name):
        super().__init__(name)
        torch = self.torch
        a, b = main_letter_ids(self.tok)
        U = self.model.lm_head.weight
        self.wlogit = ((U[a].float() - U[b].float()) * self.model.model.norm.weight.float()).detach()
        self.eps = float(getattr(self.model.model.norm, "variance_epsilon", 1e-6))
        with torch.no_grad():
            self.Mh = [(blk.self_attn.o_proj.weight.float().T @ self.wlogit).view(self.H, self.hd) for blk in self.blocks]

    def direct(self, encs, bs=8):
        """Each head's and MLP's direct contribution to logit(A) - logit(B) at the last position: (n, blocks, H + 1)."""
        torch = self.torch
        n = len(encs)
        C = np.zeros((n, self.n_layers, self.H + 1), dtype=np.float32)
        for ch in batches(n, bs):
            ids, att, _ = self._batch_roles(encs, ch)
            cap = {}
            hooks = []
            for l, blk in enumerate(self.blocks):
                hooks.append(blk.self_attn.o_proj.register_forward_pre_hook(lambda m, i, l=l: cap.__setitem__(("z", l), i[0][:, -1].float())))
                hooks.append(blk.mlp.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(("m", l), o[:, -1].float())))
            hooks.append(self.blocks[-1].register_forward_hook(
                lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[:, -1].float())))
            try:
                with torch.no_grad():
                    forward(self.model, ids, att, torch)
            finally:
                for h in hooks:
                    h.remove()
            with torch.no_grad():
                rms = torch.sqrt((cap["x"] ** 2).mean(-1) + self.eps)
                for l in range(self.n_layers):
                    z = cap[("z", l)].view(len(ch), self.H, self.hd)
                    C[ch, l, :self.H] = (torch.einsum("bhd,hd->bh", z, self.Mh[l]) / rms[:, None]).cpu().numpy()
                    C[ch, l, self.H] = ((cap[("m", l)] @ self.wlogit) / rms).cpu().numpy()
        return C


def direct_dir(M, encs, vec, n_blocks, bs=8):
    """Each head's and MLP's contribution along `vec` at the last position, blocks 0..n_blocks-1: (n, n_blocks, H + 1)."""
    torch = M.torch
    v = torch.tensor(vec, dtype=torch.float32, device=M.dev)
    with torch.no_grad():
        Mv = [(blk.self_attn.o_proj.weight.float().T @ v).view(M.H, M.hd) for blk in M.blocks[:n_blocks]]
    n = len(encs)
    C = np.zeros((n, n_blocks, M.H + 1), dtype=np.float32)
    for ch in batches(n, bs):
        ids, att, _ = M._batch_roles(encs, ch)
        cap = {}
        hooks = []
        for l, blk in enumerate(M.blocks[:n_blocks]):
            hooks.append(blk.self_attn.o_proj.register_forward_pre_hook(lambda m, i, l=l: cap.__setitem__(("z", l), i[0][:, -1].float())))
            hooks.append(blk.mlp.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(("m", l), o[:, -1].float())))
        try:
            with torch.no_grad():
                forward(M.model, ids, att, torch)
        finally:
            for h in hooks:
                h.remove()
        with torch.no_grad():
            for l in range(n_blocks):
                z = cap[("z", l)].view(len(ch), M.H, M.hd)
                C[ch, l, :M.H] = torch.einsum("bhd,hd->bh", z, Mv[l]).cpu().numpy()
                C[ch, l, M.H] = (cap[("m", l)] @ v).cpu().numpy()
    return C


def top_by_slope(Cr, Cd, eff, sel, H, k=TOP_K):
    """Pieces ranked by how much of a swap's change in the logit their own change carries (slope through zero)."""
    dC = (Cd - Cr)[sel]
    score = (dC * eff[sel][:, None, None]).sum(0) / (eff[sel] ** 2).sum()
    nb_ = Cr.shape[1]
    parts = [("head", l, h) for l in range(nb_) for h in range(H)] + [("mlp", l) for l in range(nb_)]
    sc = {p: float(score[p[1], p[2]] if p[0] == "head" else score[p[1], H]) for p in parts}
    return sorted(parts, key=lambda p: -sc[p])[:k], sc


def order_flips(L, gap=GAP, cap=None):
    rev = tot = 0
    for i in range(L.shape[0]):
        for j in range(i + 1, L.shape[0]):
            ok = np.ones(L.shape[1], bool) if cap is None else (np.abs(L[i]) < cap) & (np.abs(L[j]) < cap)
            d = (L[i] - L[j])[ok]
            big = np.abs(d) > gap
            if big.sum() >= 2:
                s = np.sign(d[big])
                tot += 1
                rev += int(s.min() != s.max())
    return rev / max(tot, 1), tot


def cycles(L, margin=GAP):
    win, lose = L > margin, L < -margin
    c = 0
    for k in range(L.shape[1]):
        for l in range(L.shape[1]):
            if k != l:
                c += int((win[:, k] & lose[:, l]).sum()) * int((lose[:, k] & win[:, l]).sum())
    return c // 2


def piece_name(p):
    return f"MLP {p[1]}" if p[0] == "mlp" else f"head {p[1]}.{p[2]}"


def random_like(top, H, rng, n_blocks=None):
    """For each top piece, another piece of its block (or, if its block has none left, of the nearest block that has)."""
    used, out = set(top), []
    for p in top:
        for dist in range(0, 1 + (n_blocks or 1000)):
            blocks = [b for b in {p[1] - dist, p[1] + dist} if b >= 0 and (n_blocks is None or b < n_blocks)]
            cs = [x for b in blocks for x in [("head", b, h) for h in range(H)] + [("mlp", b)] if x not in used]
            if cs:
                break
        x = cs[rng.integers(len(cs))]
        used.add(x)
        out.append(x)
    return out


def run(key, quick, out="real"):
    t0 = time.time()
    res = HERE / "results" / ("quick" if quick else out) / key
    res.mkdir(parents=True, exist_ok=True)
    S = p11.items(quick)
    vals = sorted({s["value"] for s in S})
    n = len(S)
    groups = np.array([vals.index(s["value"]) for s in S])
    new = [d for d in json.load(open(ROOT / "pilot13-recipe-in-words/results/real/features13.json")) if d["set"] == "new"]
    Bs = [b["text"] for b in p14.pick_partners(new, 8)] + [p15.B_TEXT]
    uB13 = [b["u"] for b in p14.pick_partners(new, 8)] + [0.35]
    nb = len(Bs)
    mid = nb - 1
    M = Runner17(MODELS[key])
    nL = M.n_layers
    layers = sorted({max(1, round(f * nL)) for f in FRACS})
    li_main = layers.index(round(0.6 * nL)) if round(0.6 * nL) in layers else len(layers) // 2
    var_key, vt = [], []
    for i, s in enumerate(S):
        w = s["text"].split(" ")
        for k in range(len(w)):
            var_key.append((i, k))
            vt.append(" ".join(w[:k] + w[k + 1:]))
    parent = np.array([i for i, _ in var_key])
    nd = len(var_key)
    texts = [s["text"] for s in S] + vt
    nt = len(texts)
    log(f"{key} ({MODELS[key]}, {nL} blocks, {M.H} heads): {n} outcomes, {nd} deletions, {nb} alternatives; layers {layers}")

    # ---- (a) fixed or relative price
    encs = [p15.encode_roles(M.tok, p11.menu_user(t, b), t, b) for b in Bs for t in texts]
    X, lg = M.read_sel(encs, layers)
    LG = lg.reshape(nb, nt).T
    L_int, L_del = LG[:n], LG[n:]
    dl = L_del - L_int[parent]
    ext = [int(np.argmin(uB13[:8])), int(np.argmax(uB13[:8]))]
    keep = [k for k in range(nb) if k not in ext]
    cs_words, cs_out = p16.common_share(dl), p16.common_share(L_int)
    flips, nflip = order_flips(L_int)
    flips8, nflip8 = order_flips(L_int, cap=8)
    cyc = cycles(L_int)
    sat = float((np.abs(L_int) > 10).mean())
    q1 = bool(cs_words >= Q1_MIN and cs_out >= Q1_MIN)
    log(f"{key} price: common share words {cs_words:.3f}, outcomes {cs_out:.3f}; order flips {flips:.2f} ({nflip} pairs; "
        f"{flips8:.2f} within ±8); cycles {cyc}; |logit| > 10 in {sat:.0%}")
    rows_int = np.array([k * nt + i for k in range(nb) for i in range(n)])
    rows_del = np.array([k * nt + n + j for k in range(nb) for j in range(nd)])
    inside = {}
    for pi, pname in enumerate(p16.POS):
        for li, L in enumerate(layers):
            pin, pdel, _ = p16.ridge_fit_predict(M.torch, M.dev, X[rows_int, pi, li].astype(np.float32), lg[rows_int], np.tile(groups, nb),
                                                 X[rows_del, pi, li].astype(np.float32))
            dp = pdel.reshape(nb, nd).T - pin.reshape(nb, n).T[parent]
            inside[f"{pname}@{L}"] = p16.common_share(dp)
    # the price before B is read (pilot 15's Q1), with the mid-priced B
    halfA = np.isin(groups[parent], np.arange(0, len(vals), 2))
    before = {}
    for pi, pname in enumerate(p16.POS[:2]):
        for li, L in enumerate(layers):
            xo = X[mid * nt + np.arange(n), pi, li].astype(np.float64)
            xv = X[mid * nt + n + np.arange(nd), pi, li].astype(np.float64)
            _, pv, po = p15.t_readout(xo, L_int[:, mid], xv, groups)
            d_ = pv - po[parent]
            before[f"{pname}@{L}"] = {"all": pb.r(d_, dl[:, mid]), "halfA": pb.r(d_[halfA], dl[halfA, mid]), "halfB": pb.r(d_[~halfA], dl[~halfA, mid])}
    best_a = max((k for k in before if k.startswith("A_nl")), key=lambda k: before[k]["halfA"])
    del X

    # ---- (b) the comparator
    a_main = L_int.mean(1)
    rng = np.random.default_rng(1700)
    donors = []
    for i in range(n):
        c = [j for j in range(n) if S[j]["value"] != S[i]["value"] and abs(a_main[j] - a_main[i]) >= GAP]
        donors.append(int(rng.choice(c)) if c else int((i + 1) % n))
    enc_r = [encs[mid * nt + i] for i in range(n)]
    enc_d = [encs[mid * nt + j] for j in donors]
    C_r, C_d = M.direct(enc_r), M.direct(enc_d)
    base_r, base_d = L_int[:, mid], L_int[donors, mid]
    eff = base_d - base_r
    sel = groups % 2 == 0
    test = np.where(~sel)[0]
    top, sc = top_by_slope(C_r, C_d, eff, sel, M.H)
    rsets = [random_like(top, M.H, np.random.default_rng(1701 + k), nL) for k in range(N_RANDOM)]
    resid_layers = sorted({max(1, round(f * nL)) for f in RESID_FRACS})
    rc = [("resid", "END", L) for L in resid_layers]
    br, bd, obs = M.patch([enc_r[i] for i in test], [enc_d[i] for i in test], rc)
    eff_t = bd - br
    T = {f"END@{c[2]}": p15.transfer(obs[ci], eff_t) for ci, c in enumerate(rc)}
    # the decision writers: pieces before the layer where the decision reaches the answer position, writing the
    # decision's direction there (as pilots 5-6 found theirs), chosen on the same half of the values
    Lstar = next((L for L in resid_layers if T[f"END@{L}"]["transfer"] >= 0.5), resid_layers[-1])
    Xs, _ = M.read_sel(enc_r, [Lstar])
    vdec = pb.dom_direction(Xs[:, p16.POS.index("END"), 0].astype(np.float64), base_r)
    W_r, W_d = direct_dir(M, enc_r, vdec, Lstar), direct_dir(M, enc_d, vdec, Lstar)
    topw, scw = top_by_slope(W_r, W_d, eff, sel, M.H)
    rsetsw = [random_like(topw, M.H, np.random.default_rng(1711 + k), Lstar) for k in range(N_RANDOM)]
    sets = {"logit_writers": top, **{f"logit_random_{k + 1}": r for k, r in enumerate(rsets)},
            "decision_writers": topw, **{f"decision_random_{k + 1}": r for k, r in enumerate(rsetsw)}}
    p5p = None
    if key in ("qwen14",):
        p5p = [p15.parse_piece(x) for x in json.load(open(ROOT / "pilot5-value-circuit/results/real/patching.json"))["pieces"]["top-20"]]
        sets["pilot5_pieces"] = p5p
    pc = [("pieces", v) for v in sets.values()]
    _, _, obp = M.patch([enc_r[i] for i in test], [enc_d[i] for i in test], pc)
    sentT = {k: p15.transfer(obp[ci], eff_t) for ci, k in enumerate(sets)}
    top_t, rand_t = sentT["logit_writers"], [sentT[f"logit_random_{k + 1}"]["transfer"] for k in range(N_RANDOM)]
    topw_t, randw_t = sentT["decision_writers"], [sentT[f"decision_random_{k + 1}"]["transfer"] for k in range(N_RANDOM)]
    q2 = bool(top_t["transfer"] >= Q2_MIN and max(rand_t) <= RANDOM_MAX)
    q2b = bool(topw_t["transfer"] >= Q2B_MIN and max(randw_t) <= RANDOM_MAX)
    log(f"{key} comparator: decision reaches the answer position at layer {Lstar}; logit writers carry {top_t['transfer']:.2f} "
        f"(random {', '.join(f'{x:.2f}' for x in rand_t)}); decision writers (blocks < {Lstar}) carry {topw_t['transfer']:.2f} "
        f"(random {', '.join(f'{x:.2f}' for x in randw_t)}); answer position: " + ", ".join(f"{k} {v['transfer']:.2f}" for k, v in T.items()))
    P6 = [json.loads(l) for l in open(ROOT / "pilot6-value-subspace/pairs.jsonl")]
    generic = {}
    for task in ("quantity", "weight"):
        rows = [r for r in P6 if r["set"] == task]
        rows = [rows[i] for i in np.random.default_rng(1702).choice(len(rows), size=min(N_CONTROL if not quick else 24, len(rows)), replace=False)]
        er = [encode_plain(M.tok, r["user"]) for r in rows]
        ed = [encode_plain(M.tok, r["donor_user"]) for r in rows]
        br_, bd_, ob_ = M.patch(er, ed, rc + pc)
        e_ = bd_ - br_
        tr_ = {k: p15.transfer(ob_[len(rc) + ci], e_)["transfer"] for ci, k in enumerate(sets)}
        generic[task] = {"top": tr_["logit_writers"], "random": [tr_[f"logit_random_{k + 1}"] for k in range(N_RANDOM)],
                         "decision_writers": tr_["decision_writers"], "decision_random": [tr_[f"decision_random_{k + 1}"] for k in range(N_RANDOM)],
                         "pilot5_pieces": tr_.get("pilot5_pieces"),
                         "answer_position": {f"END@{c[2]}": p15.transfer(ob_[ci], e_)["transfer"] for ci, c in enumerate(rc)},
                         "n_pairs": len(rows), "mean_abs_effect": float(np.abs(e_).mean())}
        log(f"{key} {task}: logit writers {generic[task]['top']:.2f}, decision writers {generic[task]['decision_writers']:.2f} "
            f"(random {', '.join(f'{x:.2f}' for x in generic[task]['decision_random'])})")
    q3 = bool(all(generic[t]["top"] >= Q3_MIN and max(generic[t]["random"]) <= RANDOM_MAX for t in generic))
    q3b = bool(all(generic[t]["decision_writers"] >= Q3_MIN and max(generic[t]["decision_random"]) <= RANDOM_MAX for t in generic))
    p5 = None
    if p5p is not None:
        p5 = {"sentences": sentT["pilot5_pieces"]["transfer"], "overlap_with_top20": len(set(p5p) & set(top)),
              "overlap_with_decision_writers": len(set(p5p) & set(topw)), "method_check_pass": len(set(p5p) & set(topw)) >= 5,
              "apples": generic["quantity"]["pilot5_pieces"], "weight": generic["weight"]["pilot5_pieces"]}
        log(f"{key} method check: {p5['overlap_with_decision_writers']} of pilot 5's 20 pieces among the decision writers "
            f"(logit writers: {p5['overlap_with_top20']}); pilot 5's pieces carry {p5['sentences']:.2f}")

    V = {"model": MODELS[key], "blocks": nL, "heads": M.H, "layers_read": layers, "B": Bs,
         "Q1": {"common_share_words": cs_words, "common_share_outcomes": cs_out, "pass": q1},
         "Q2": {"top_transfer": top_t["transfer"], "ci95": top_t["ci95"], "random": rand_t, "pass": q2},
         "Q3": {"apples": generic["quantity"]["top"], "weight": generic["weight"]["top"],
                "random_apples": generic["quantity"]["random"], "random_weight": generic["weight"]["random"], "pass": q3},
         "decision_layer": Lstar,
         "Q2b": {"transfer": topw_t["transfer"], "ci95": topw_t["ci95"], "random": randw_t, "pass": q2b},
         "Q3b": {"apples": generic["quantity"]["decision_writers"], "weight": generic["weight"]["decision_writers"],
                 "random_apples": generic["quantity"]["decision_random"], "random_weight": generic["weight"]["decision_random"], "pass": q3b},
         "decision_writers": [piece_name(p) + f" ({scw[p]:+.3f})" for p in topw],
         "without_extreme_Bs": {"words": p16.common_share(dl[:, keep]), "outcomes": p16.common_share(L_int[:, keep])},
         "order_flips": flips, "order_flip_pairs": nflip, "order_flips_within_8": flips8, "cycles_margin2": cyc, "share_logits_beyond_10": sat,
         "inside_common_share": inside, "price_before_B": before, "price_before_B_best": {"where": best_a, "r": before[best_a]["halfB"]},
         "answer_position_sentences": {k: v["transfer"] for k, v in T.items()}, "generic": generic,
         "top20": [piece_name(p) + f" ({sc[p]:+.3f})" for p in top], "random_sets": [[piece_name(p) for p in r] for r in rsets],
         "pilot5_pieces": p5, "minutes": (time.time() - t0) / 60}
    json.dump(V, open(res / "verdicts17.json", "w"), indent=1)
    np.savez_compressed(res / "effects17.npz", logits_intact=L_int, logits_deleted=L_del, dl=dl, C_r=C_r, C_d=C_d, W_r=W_r, W_d=W_d,
                        donors=np.array(donors), vdec=vdec.astype(np.float32))
    log(f"{key} done in {V['minutes']:.1f} min: Q1 {'fixed' if q1 else 'relative'}, Q2 {q2}, Q3 {q3}, Q2b {q2b}, Q3b {q3b}")
    del M
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return V


def report(Vs, res):
    ok = lambda b: "yes" if b else "no"
    rows = ["# Pilot 17 results", "", "## The questions, per model", "",
            "| | " + " | ".join(f"{k} ({v['model'].split('/')[-1]})" for k, v in Vs.items()) + " |", "|---" * (len(Vs) + 1) + "|"]
    get = lambda f: " | ".join(f(v) for v in Vs.values())
    rows += [
        "| **Q1** fixed price? (words / outcomes; bar 0.9) | " + get(lambda v: f"{ok(v['Q1']['pass'])}: {v['Q1']['common_share_words']:.2f} / {v['Q1']['common_share_outcomes']:.2f}") + " |",
        "| **Q2** compact comparator? (top 20 on held-out sentence swaps; bar 0.5) | " + get(lambda v: f"{ok(v['Q2']['pass'])}: {v['Q2']['top_transfer']:.2f} (random ≤ {max(v['Q2']['random']):.2f})") + " |",
        "| **Q3** generic? (apples / weight; bar 0.3) | " + get(lambda v: f"{ok(v['Q3']['pass'])}: {v['Q3']['apples']:.2f} / {v['Q3']['weight']:.2f}") + " |",
        "| layer where the decision reaches the answer position (whole-state swap ≥ 0.5) | " + get(lambda v: str(v.get("decision_layer", ""))) + " |",
        "| **Q2b** decision writers before that layer (held-out sentence swaps; bar 0.4) | " + get(lambda v: "" if "Q2b" not in v else f"{ok(v['Q2b']['pass'])}: {v['Q2b']['transfer']:.2f} (random ≤ {max(v['Q2b']['random']):.2f})") + " |",
        "| **Q3b** the same on apples / weight (bar 0.3) | " + get(lambda v: "" if "Q3b" not in v else f"{ok(v['Q3b']['pass'])}: {v['Q3b']['apples']:.2f} / {v['Q3b']['weight']:.2f} (random ≤ {max(v['Q3b']['random_apples'] + v['Q3b']['random_weight']):.2f})") + " |",
        "", "## Also", "",
        "| | " + " | ".join(Vs) + " |", "|---" * (len(Vs) + 1) + "|",
        "| without the two extreme alternatives (words / outcomes) | " + get(lambda v: f"{v['without_extreme_Bs']['words']:.2f} / {v['without_extreme_Bs']['outcomes']:.2f}") + " |",
        "| order flips (clear pairs; within ±8 logits) | " + get(lambda v: f"{v['order_flips']:.2f} ({v['order_flip_pairs']}); {v['order_flips_within_8']:.2f}") + " |",
        "| cycles (all gaps > 2 logits) | " + get(lambda v: str(v["cycles_margin2"])) + " |",
        "| menus with \\|logit\\| > 10 | " + get(lambda v: f"{v['share_logits_beyond_10']:.0%}") + " |",
        "| inside: common share at the line break after B, by layer | " + get(lambda v: ", ".join(f"{k.split('@')[1]}: {x:.2f}" for k, x in v["inside_common_share"].items() if k.startswith("B_nl"))) + " |",
        "| check: line break after A (must be 1) | " + get(lambda v: ", ".join(f"{x:.3f}" for k, x in v["inside_common_share"].items() if k.startswith("A_nl"))) + " |",
        "| price before B is read: best line break after A (held out) | " + get(lambda v: f"{v['price_before_B_best']['where']} r {v['price_before_B_best']['r']:.2f}") + " |",
        "| price at the line break after B (mid B), by layer | " + get(lambda v: ", ".join(f"{k.split('@')[1]}: {x['all']:.2f}" for k, x in v["price_before_B"].items() if k.startswith("B_nl"))) + " |",
        "| decision at the answer position: whole-state swap by layer (sentences) | " + get(lambda v: ", ".join(f"{k.split('@')[1]}: {x:.2f}" for k, x in v["answer_position_sentences"].items())) + " |",
        "| the same for apples | " + get(lambda v: ", ".join(f"{k.split('@')[1]}: {x:.2f}" for k, x in v["generic"]["quantity"]["answer_position"].items())) + " |",
        "| top 20 pieces | " + get(lambda v: "; ".join(v["top20"][:8]) + " …") + " |",
        "| pilot 5's 20 pieces (qwen14 only): transfer on sentences / apples / weight; shared with the logit writers / decision writers | " + get(lambda v: "" if not v["pilot5_pieces"] else f"{v['pilot5_pieces']['sentences']:.2f} / {v['pilot5_pieces'].get('apples') or 0:.2f} / {v['pilot5_pieces'].get('weight') or 0:.2f}; {v['pilot5_pieces']['overlap_with_top20']} / {v['pilot5_pieces'].get('overlap_with_decision_writers', '')}") + " |",
        "| decision writers | " + get(lambda v: "; ".join(v.get("decision_writers", [])[:8]) + " …") + " |",
        "| run time (min) | " + get(lambda v: f"{v['minutes']:.1f}") + " |"]
    (res / "report.md").write_text("\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="comma-separated keys: " + ", ".join(MODELS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="real")
    a = ap.parse_args()
    res = HERE / "results" / ("quick" if a.quick else a.out)
    Vs = {}
    for key in a.model.split(","):
        try:
            Vs[key] = run(key, a.quick, a.out)
        except Exception as e:  # keep going with the next model
            import traceback
            traceback.print_exc()
            log(f"{key} FAILED: {e!r}")
        if Vs:
            report(Vs, res)
            json.dump(Vs, open(res / "verdicts17.json", "w"), indent=1)
        if not a.quick:                     # free the disk for the next model
            import os
            import shutil
            hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
            shutil.rmtree(hf / "hub" / ("models--" + MODELS[key].replace("/", "--")), ignore_errors=True)
            shutil.rmtree(hf / "xet", ignore_errors=True)       # the chunk cache (pilot 17's first run filled the disk)


if __name__ == "__main__":
    main()
