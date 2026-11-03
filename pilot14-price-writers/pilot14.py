"""Pilot 14: what writes the price (DESIGN.md).

  python pilot14.py            Qwen3-14B: results/real/
  python pilot14.py --quick    Qwen3-0.6B on 6 values (24 sentences): results/quick/ (a pipeline test)

Steps: the price direction v at layer 20 (as pilots 11-13); the exact split of the readout into the embedding and every
head and MLP of blocks 1-20 (and the means used for ablation); the choice's price test against 8 fixed partners (pilot
13's new sentences); word deletions (Q1, Q2); the top parts and their ablation against matched random sets (Q3, Q4);
the block-by-block ablation map; what the top heads attend to; the report.
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
          "pilot11-vibe-direction", "pilot12-recipes", "pilot13-recipe-in-words", "pilot14-price-writers"):
    sys.path.insert(0, str(ROOT / p))
import pilot11 as p11  # noqa: E402
import probe11 as pb  # noqa: E402
from common import batches, forward, log  # noqa: E402

L_READ = 20                     # the readout layer (pilots 11-13)
N_MAP = 24                      # the block-by-block map covers blocks 1-24
TOP_K, N_RANDOM_SETS, MLP_WINDOW = 10, 5, 3
Q1_MIN, Q2_MIN, Q3_MIN, Q4_MIN = 0.5, 0.6, 0.5, 0.3
N = p11.N_PEOPLE
MARK = f"A) {N} people "


def encode_words(tok, user, marker, text):
    """As pilot 11's encode, plus the word (index in text.split(' ')) that each of the sentence's tokens belongs to."""
    s = p11.p7.chat_text(tok, user)
    e = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    start = s.index(user) + user.index(marker) + len(marker)
    end = start + len(text)
    span = [i for i, (x, y) in enumerate(e["offset_mapping"]) if y > start and x < end and y > x]
    assert span and s[start:end] == text, (user, text)
    bounds, pos = [], start
    for w in text.split(" "):
        bounds.append((pos, pos + len(w)))
        pos += len(w) + 1
    word_of = []
    for i in span:
        x, y = e["offset_mapping"][i]
        word_of.append(int(np.argmax([min(y, b) - max(x, a) for a, b in bounds])))
    return e["input_ids"], span, word_of


class Runner14(p11.Runner):
    def __init__(self, name):
        super().__init__(name)
        c = self.model.config
        self.H = c.num_attention_heads
        self.hd = getattr(c, "head_dim", None) or c.hidden_size // self.H

    def attribute(self, encs, v, L, n_mean, bs=8):
        """For each sentence, averaged over its tokens: every head's and MLP's contribution along v in blocks 1..L, the
        embedding's, and the readout itself (block L's output along v), also token by token. And the mean input of every
        head to o_proj and the mean MLP output over all sentences' tokens in blocks 1..n_mean (for mean ablation)."""
        torch = self.torch
        vt = torch.tensor(v, dtype=torch.float32, device=self.dev)
        U = [(blk.self_attn.o_proj.weight.float().T @ vt).view(self.H, self.hd) for blk in self.blocks[:L]]
        n, top = len(encs), max(L, n_mean)
        heads, mlps, emb, readout, tokens = np.zeros((n, L, self.H)), np.zeros((n, L)), np.zeros(n), np.zeros(n), [None] * n
        zsum = torch.zeros((n_mean, self.H * self.hd), device=self.dev)
        msum = torch.zeros((n_mean, self.d), device=self.dev)
        cnt = 0
        for ch in batches(n, bs):
            ids, att, spans = self._batch(encs, ch)
            cap = {}
            hooks = [self.model.model.embed_tokens.register_forward_hook(lambda m, i, o: cap.__setitem__("emb", o))]
            for l in range(top):
                blk = self.blocks[l]
                hooks.append(blk.self_attn.o_proj.register_forward_pre_hook(lambda m, i, l=l: cap.__setitem__(("z", l), i[0])))
                hooks.append(blk.mlp.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(("m", l), o)))
            hooks.append(self.blocks[L - 1].register_forward_hook(lambda m, i, o: cap.__setitem__("res", o[0] if isinstance(o, tuple) else o)))
            try:
                with torch.no_grad():
                    self.model.model(input_ids=ids, attention_mask=att)
            finally:
                for h in hooks:
                    h.remove()
            with torch.no_grad():
                for j, i in enumerate(ch):
                    sp = torch.tensor(spans[j], device=self.dev)
                    emb[i] = (cap["emb"][j, sp].float() @ vt).mean().item()
                    rt = cap["res"][j, sp].float() @ vt
                    readout[i], tokens[i] = rt.mean().item(), rt.cpu().numpy()
                    for l in range(L):
                        z = cap[("z", l)][j, sp].float().view(len(sp), self.H, self.hd)
                        heads[i, l] = (torch.einsum("thd,hd->h", z, U[l]) / len(sp)).cpu().numpy()
                        mlps[i, l] = (cap[("m", l)][j, sp].float() @ vt).mean().item()
                    for l in range(n_mean):
                        zsum[l] += cap[("z", l)][j, sp].float().sum(0)
                        msum[l] += cap[("m", l)][j, sp].float().sum(0)
                    cnt += len(sp)
        return dict(heads=heads, mlps=mlps, emb=emb, readout=readout, tokens=tokens, zmean=zsum / cnt, mmean=msum / cnt)

    def _ablate(self, abl, spans):
        """Mean ablation at the sentence's tokens: abl = {"heads": {block: [head, ...]}, "mlps": [block, ...],
        "zmean": ..., "mmean": ...} (blocks counted from 0)."""
        torch = self.torch
        sps = [torch.tensor(s, device=self.dev) for s in spans]
        hooks = []
        for l, hs in abl.get("heads", {}).items():
            cols = torch.cat([torch.arange(h * self.hd, (h + 1) * self.hd) for h in hs]).to(self.dev)
            zm = abl["zmean"][l, cols]

            def pre(m, inp, cols=cols, zm=zm):
                z = inp[0].clone()
                for j, sp in enumerate(sps):
                    z[j, sp[:, None], cols[None, :]] = zm.to(z.dtype)
                return (z,) + tuple(inp[1:])
            hooks.append(self.blocks[l].self_attn.o_proj.register_forward_pre_hook(pre))
        for l in abl.get("mlps", []):
            mm = abl["mmean"][l]

            def fwd(m, i, o, mm=mm):
                o = o.clone()
                for j, sp in enumerate(sps):
                    o[j, sp] = mm.to(o.dtype)
                return o
            hooks.append(self.blocks[l].mlp.register_forward_hook(fwd))
        return hooks

    def menu_logit_abl(self, encs, abl=None, bs=32):
        """logit P(A), with an optional mean ablation at option A's sentence tokens."""
        torch = self.torch
        out = np.zeros(len(encs))
        for ch in batches(len(encs), bs):
            ids, att, spans = self._batch(encs, ch)
            hooks = self._ablate(abl, spans) if abl else []
            try:
                lg = forward(self.model, ids, att, torch).logits[:, -1].float()
            finally:
                for h in hooks:
                    h.remove()
            lp = torch.log_softmax(lg, -1)
            out[ch] = (torch.logsumexp(lp[:, self.AB[0]], -1) - torch.logsumexp(lp[:, self.AB[1]], -1)).cpu().numpy()
        return out

    def attend(self, encs, heads, bs=8):
        """For each (block, head): its attention from the sentence's tokens (mean over them) to each of the sentence's tokens."""
        torch = self.torch
        out = {lh: [None] * len(encs) for lh in heads}
        layers = sorted({l for l, _ in heads})
        for ch in batches(len(encs), bs):
            ids, att, spans = self._batch(encs, ch)
            cap = {}
            hooks = [self.blocks[l].self_attn.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, o[1])) for l in layers]
            try:
                with torch.no_grad():
                    self.model.model(input_ids=ids, attention_mask=att)
            finally:
                for h in hooks:
                    h.remove()
            for (l, h) in heads:
                for j, i in enumerate(ch):
                    sp = torch.tensor(spans[j], device=self.dev)
                    out[(l, h)][i] = cap[l][j, h][sp][:, sp].float().mean(0).cpu().numpy()
        return out


def slope(m, u):
    return float(np.polyfit(u, m, 1)[0])


def pick_partners(new, k):
    """k of pilot 13's new sentences, nearest to evenly spaced quantiles of their prices."""
    un = np.array([d["u"] for d in new])
    chosen = []
    for q in (np.arange(k) + 0.5) / k:
        t = np.quantile(un, q)
        j = min((j for j in range(len(new)) if j not in chosen), key=lambda j: abs(un[j] - t))
        chosen.append(j)
    return [new[j] for j in chosen]


def matched_random(top, n_layers_avail, H, rng, taken):
    """A random set matched to `top` (list of ("head", l, h) / ("mlp", l)): heads from the same block, MLPs from a block
    within ±MLP_WINDOW; none from `taken` and no repeats."""
    used, out = set(taken), []
    for part in top:
        if part[0] == "head":
            l = part[1]
            cands = [("head", l, h) for h in range(H) if ("head", l, h) not in used]
        else:
            l = part[1]
            cands = []
            for w in (MLP_WINDOW, 6, n_layers_avail):
                cands = [("mlp", k) for k in range(max(0, l - w), min(n_layers_avail, l + w + 1)) if ("mlp", k) not in used]
                if cands:
                    break
        c = cands[rng.integers(len(cands))]
        used.add(c)
        out.append(c)
    return out


def as_abl(parts, means):
    heads, mlps = {}, []
    for p in parts:
        if p[0] == "head":
            heads.setdefault(p[1], []).append(p[2])
        else:
            mlps.append(p[1])
    return {"heads": heads, "mlps": mlps, **means}


def name(p):
    return f"head {p[1] + 1}.{p[2]}" if p[0] == "head" else f"MLP {p[1] + 1}"


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
    new = [d for d in json.load(open(ROOT / "pilot13-recipe-in-words/results/real/features13.json")) if d["set"] == "new"]
    P8 = pick_partners(new, 8)
    P4 = P8[::2]
    M = Runner14(model_name)
    log(f"{model_name}: {n} sentences; partners " + ", ".join(f"{d['u']:+.1f}" for d in P8))

    # ---- 1. the price direction at layer 20 (as pilots 11-13), then the exact split of its readout
    enc_A = [encode_words(M.tok, p11.menu_user(s["text"], P8[0]["text"]), MARK, s["text"]) for s in S]
    X, _ = M.read([(e[0], e[1]) for e in enc_A], [L_READ])
    v, _, c = pb.full_dom(X[:, 0].astype(np.float64), u)
    A = M.attribute([(e[0], e[1]) for e in enc_A], v, L_READ, N_MAP)
    total = A["emb"] + A["heads"].sum((1, 2)) + A["mlps"].sum(1)
    ident_r, ident_err = pb.r(total, A["readout"]), float(np.abs(total - A["readout"]).max() / A["readout"].std())
    var = np.var(total)
    cov = lambda x: float(np.mean((x - x.mean()) * (total - total.mean())) / var)
    share_heads = np.array([[cov(A["heads"][:, l, h]) for h in range(M.H)] for l in range(L_READ)])
    share_mlps = np.array([cov(A["mlps"][:, l]) for l in range(L_READ)])
    share_emb = cov(A["emb"])
    parts = [("head", l, h) for l in range(L_READ) for h in range(M.H)] + [("mlp", l) for l in range(L_READ)]
    share = {p: (share_heads[p[1], p[2]] if p[0] == "head" else share_mlps[p[1]]) for p in parts}
    top = sorted(parts, key=lambda p: -share[p])[:TOP_K]
    q3 = float(sum(share[p] for p in top))
    log(f"attribution: identity r {ident_r:.4f} (max error {ident_err:.3f} SD); embedding share {share_emb:+.2f}; "
        f"top {TOP_K}: " + ", ".join(f"{name(p)} {share[p]:+.2f}" for p in top) + f"; Q3 {q3:.2f}")

    # ---- 2. the choice's price test: each sentence as A against the 8 partners, clean
    menu8 = [p11.encode(M.tok, p11.menu_user(s["text"], d["text"]), MARK, s["text"]) for s in S for d in P8]
    menu4 = [p11.encode(M.tok, p11.menu_user(s["text"], d["text"]), MARK, s["text"]) for s in S for d in P4]
    m8 = lambda abl=None: M.menu_logit_abl(menu8, abl).reshape(n, len(P8)).mean(1)
    m4 = lambda abl=None: M.menu_logit_abl(menu4, abl).reshape(n, len(P4)).mean(1)
    clean8 = m8()
    clean4 = m4()
    b8, b4 = slope(clean8, u), slope(clean4, u)
    log(f"choice test: slope of the mean logit on the price {b8:.3f} (8 partners), {b4:.3f} (4); r {pb.r(clean8, u):.2f}")

    # ---- 3. Q4: ablate the top parts, and 5 matched random sets
    means = {"zmean": A["zmean"], "mmean": A["mmean"]}
    lost_top = 1 - slope(m8(as_abl(top, means)), u) / b8
    rng = np.random.default_rng(1400)
    rand_sets = [matched_random(top, L_READ, M.H, rng, top) for _ in range(N_RANDOM_SETS)]
    lost_rand = [1 - slope(m8(as_abl(rs, means)), u) / b8 for rs in rand_sets]
    q4 = bool(lost_top >= Q4_MIN and lost_top > max(lost_rand))
    log(f"Q4: lost {lost_top:.2f} with the top {TOP_K}; random sets " + ", ".join(f"{x:.2f}" for x in lost_rand))

    # ---- 4. words: delete each word in turn
    var_enc, var_key = [], []
    for i, s in enumerate(S):
        w = s["text"].split(" ")
        for k in range(len(w)):
            t = " ".join(w[:k] + w[k + 1:])
            var_key.append((i, k))
            var_enc.append([p11.encode(M.tok, p11.menu_user(t, d["text"]), MARK, t) for d in P8])
    flat = [e for es in var_enc for e in es]
    vm = M.menu_logit_abl(flat).reshape(len(var_enc), len(P8)).mean(1)
    Xv, _ = M.read([es[0] for es in var_enc], [L_READ])
    rv = Xv[:, 0].astype(np.float64) @ v
    ro = X[:, 0].astype(np.float64) @ v
    d_price = np.array([vm[j] - clean8[i] for j, (i, k) in enumerate(var_key)])
    d_read = np.array([rv[j] - ro[i] for j, (i, k) in enumerate(var_key)])
    q2 = pb.r(d_read, d_price)
    top2 = []
    words = []
    for i, s in enumerate(S):
        idx = [j for j, (ii, k) in enumerate(var_key) if ii == i]
        e = np.abs(d_price[idx])
        top2.append(float(np.sort(e)[-2:].sum() / e.sum()))
        w = s["text"].split(" ")
        words.append({"value": s["value"], "text": s["text"], "u": float(u[i]), "words": w,
                      "contribution": [float(-d_price[j]) for j in idx], "readout_contribution": [float(-d_read[j] * c) for j in idx]})
    q1 = float(np.median(top2))
    log(f"words: Q1 median top-2 share {q1:.2f}; Q2 r {q2:.2f} over {len(var_key)} deletions")

    # ---- 5. the block-by-block map (exploratory): all heads, or the MLP, of one block at A's tokens
    bmap = []
    for l in range(min(N_MAP, M.n_layers)):
        lh = 1 - slope(m4(as_abl([("head", l, h) for h in range(M.H)], means)), u) / b4
        lm = 1 - slope(m4(as_abl([("mlp", l)], means)), u) / b4
        bmap.append({"block": l + 1, "heads_lost": float(lh), "mlp_lost": float(lm),
                     "heads_share": float(share_heads[l].sum()) if l < L_READ else None, "mlp_share": float(share_mlps[l]) if l < L_READ else None})
    log("block map: " + ", ".join(f"{b['block']}: {b['heads_lost']:+.2f}/{b['mlp_lost']:+.2f}" for b in bmap))

    # ---- 6. what the top heads read (exploratory)
    th = [(p[1], p[2]) for p in top if p[0] == "head"]
    read = {}
    if th:
        att = M.attend([(e[0], e[1]) for e in enc_A], th)
        for (l, h) in th:
            aw, cw = [], []
            for i, e in enumerate(enc_A):
                wo = np.array(e[2])
                per_word = np.array([att[(l, h)][i][wo == k].sum() for k in range(len(words[i]["words"]))])
                aw += list(per_word)
                cw += list(np.abs(words[i]["contribution"]))
            r_aw = pb.r(np.array(aw), np.array(cw))
            favourite = {}
            for i, e in enumerate(enc_A):
                wo = np.array(e[2])
                per_word = [att[(l, h)][i][wo == k].sum() for k in range(len(words[i]["words"]))]
                favourite.setdefault(words[i]["words"][int(np.argmax(per_word))].strip(",.").lower(), 0)
                favourite[words[i]["words"][int(np.argmax(per_word))].strip(",.").lower()] += 1
            read[name(("head", l, h))] = {"r_attention_with_word_effect": r_aw,
                                          "most_attended_words": sorted(favourite.items(), key=lambda x: -x[1])[:8]}

    # ---- 7. save and report
    running = []
    for i, e in enumerate(enc_A):
        toks = [M.tok.decode([e[0][t]]) for t in e[1]]
        running.append({"text": S[i]["text"], "u": float(u[i]), "tokens": toks, "readout_price_units": [float(x * c) for x in A["tokens"][i]]})
    ok = lambda b: "pass" if b else "fail"
    V = {"model": model_name, "n": n, "partners": [d["text"] for d in P8], "partner_prices": [d["u"] for d in P8],
         "readout_layer": L_READ, "price_units_per_readout_unit": float(c), "identity_r": ident_r, "identity_max_error_sd": ident_err,
         "Q1": {"median_top2_share": q1, "pass": bool(q1 >= Q1_MIN)},
         "Q2": {"r": q2, "pass": bool(q2 >= Q2_MIN), "n_deletions": len(var_key)},
         "Q3": {"top_share": q3, "pass": bool(q3 >= Q3_MIN), "top": [{"part": name(p), "share": float(share[p])} for p in top]},
         "Q4": {"lost_top": float(lost_top), "lost_random": [float(x) for x in lost_rand], "pass": q4,
                "random_sets": [[name(p) for p in rs] for rs in rand_sets]},
         "choice_slope_8": b8, "choice_slope_4": b4, "choice_r_with_price": pb.r(clean8, u),
         "embedding_share": share_emb, "heads_share_total": float(share_heads.sum()), "mlps_share_total": float(share_mlps.sum()),
         "block_map": bmap, "top_heads_read": read, "minutes": (time.time() - t0) / 60}
    json.dump(V, open(res / "verdicts14.json", "w"), indent=1)
    json.dump(words, open(res / "words14.json", "w"), indent=1)
    json.dump(running, open(res / "running14.json", "w"), indent=1)
    np.savez_compressed(res / "attribution14.npz", heads=A["heads"].astype(np.float32), mlps=A["mlps"].astype(np.float32), emb=A["emb"],
                        readout=A["readout"], u=u, v=v.astype(np.float32), share_heads=share_heads, share_mlps=share_mlps,
                        clean8=clean8, clean4=clean4, d_price=d_price, d_read=d_read, var_key=np.array(var_key))
    allw = sorted(((abs(cw), cw, s["value"], s["words"][k], s["text"]) for s in words for k, cw in enumerate(s["contribution"])), reverse=True)
    rep = [f"# Pilot 14 results ({model_name})", "",
           f"{n} sentences as option A against 8 fixed partners (pilot 13's new sentences, prices "
           + ", ".join(f"{d['u']:+.1f}" for d in P8) + f"). The choice's price slope (mean logit on price): {b8:.2f}; r {pb.r(clean8, u):.2f}. "
           f"Readout: the price direction at layer {L_READ}; 1 readout unit = {c:.3f} price units. "
           f"The split is exact up to rounding: r {ident_r:.4f} with the readout, largest error {ident_err:.3f} SD.", "",
           "## Pre-registered questions", "",
           f"- **Q1** a few words set the price: median share of the two largest word-deletion effects {q1:.2f}: **{ok(q1 >= Q1_MIN)}** (bar {Q1_MIN})",
           f"- **Q2** the readout follows the choice word by word: r {q2:.2f} over {len(var_key)} deletions: **{ok(q2 >= Q2_MIN)}** (bar {Q2_MIN})",
           f"- **Q3** a few parts write the price: the top {TOP_K} parts' summed share {q3:.2f}: **{ok(q3 >= Q3_MIN)}** (bar {Q3_MIN})",
           f"- **Q4** the choice uses them: mean-ablating them at the sentence loses {lost_top:.0%} of the choice's price slope; "
           f"the 5 matched random sets lose " + ", ".join(f"{x:.0%}" for x in lost_rand) + f": **{ok(q4)}** (≥ {Q4_MIN:.0%} and above all 5)", "",
           "## The writers", "",
           f"Shares of the readout's spread: embedding {share_emb:+.2f}, all heads {share_heads.sum():+.2f}, all MLPs {share_mlps.sum():+.2f}.", "",
           "| part | share |", "|---|---|"] + [f"| {name(p)} | {share[p]:+.3f} |" for p in top] + [
           "", "Random sets (Q4): " + "; ".join(", ".join(name(p) for p in rs) for rs in rand_sets), "",
           "## Block by block (exploratory; 4 partners)", "",
           "| block | heads: share | heads: slope lost when ablated | MLP: share | MLP: slope lost when ablated |", "|---|---|---|---|---|"]
    for b in bmap:
        hs = f"{b['heads_share']:+.3f}" if b["heads_share"] is not None else ""
        ms = f"{b['mlp_share']:+.3f}" if b["mlp_share"] is not None else ""
        rep.append(f"| {b['block']} | {hs} | {b['heads_lost']:+.2f} | {ms} | {b['mlp_lost']:+.2f} |")
    rep += ["", "## The words that move the price most (deleting the word lowers the price by this much, in price units)", "",
            "| value | word | contribution | sentence |", "|---|---|---|---|"]
    rep += [f"| {vl} | {w} | {cw:+.2f} | {t} |" for _, cw, vl, w, t in allw[:30]]
    if read:
        rep += ["", "## What the top heads read (exploratory)", "", "| head | r of attention to a word with its deletion effect | most attended words (count of sentences) |",
                "|---|---|---|"]
        rep += [f"| {k} | {x['r_attention_with_word_effect']:+.2f} | " + ", ".join(f"{w} ({cnt})" for w, cnt in x["most_attended_words"]) + " |"
                for k, x in read.items()]
    rep += ["", "## The running price (readout token by token, price units), six sentences", ""]
    for i in list(np.argsort(u)[[0, n // 4, n // 2, 3 * n // 4, -2, -1]]):
        r_ = running[i]
        rep.append(f"- {r_['text']} (price {r_['u']:+.1f}): " + " ".join(f"{t.strip()}[{x:+.1f}]" for t, x in zip(r_["tokens"], r_["readout_price_units"])))
    rep += ["", f"Run time {V['minutes']:.1f} min."]
    (res / "report.md").write_text("\n".join(rep) + "\n")
    log(f"report written; {V['minutes']:.1f} min")


if __name__ == "__main__":
    main()
