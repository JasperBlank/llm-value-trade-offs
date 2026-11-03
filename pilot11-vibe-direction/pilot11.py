"""Pilot 11: one direction for both? (DESIGN.md). Reads the states, finds the vibe direction, pushes along it.

  python pilot11.py            Qwen3-14B: results/real/
  python pilot11.py --quick    Qwen3-0.6B on the first 6 values (24 sentences): results/quick/ (a pipeline test)

Stages (all in one run, since the push needs the direction found from the states):
  read   the residual stream after blocks 0, 4, 8, ... for each sentence, in the menu context (as option A, before B)
         and the rating context (pilot 10's W0 prompt), averaged over the sentence's own tokens and at its last token
  probe  held-out readouts (probe11.py): Q1, Q2 and everything in DESIGN.md section 5
  steer  push along the direction over the sentence's tokens (menu: as option A against its 4 nearest-priced partners
         of other values; rating: its W0 prompt), and along a random direction (Q3)
  report report.md, verdicts11.json
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
for p in ("pilot7-value-route", "pilot5-value-circuit", "pilot9-vibe-check", "pilot8-intrinsic-values", "pilot10-full-range", "pilot11-vibe-direction"):
    sys.path.insert(0, str(ROOT / p))
import pilot7 as p7  # noqa: E402
import probe11 as pb  # noqa: E402
from build_prompts import WORDINGS  # noqa: E402  (pilot 9)
from build_prompts10 import BENEFICIARY, Q2, R2  # noqa: E402
from common import batches, forward, load_model, log  # noqa: E402
from pilot8 import letter_set  # noqa: E402
from pilot9_logscale import log_scale_score  # noqa: E402

BETA = 0.327          # pilot 10: rating points (W0, 8 people) per price unit, across the 96 sentences
PUSH = 2.0            # price units
SELECT = (4, 32)      # Q1's layer is chosen among these layers (inclusive)
Q1_MIN, Q2_R_MIN, Q2_PARTIAL_MAX, Q3_MIN, Q3_RANDOM_MAX = 0.7, 0.6, 0.3, 0.5, 0.1
N_PEOPLE = 8


def menu_user(a, b):
    return "\n".join([BENEFICIARY, Q2, f"A) {N_PEOPLE} people {a}", f"B) {N_PEOPLE} people {b}", R2])


def rating_user(a):
    return "\n".join([BENEFICIARY, WORDINGS[0][0], f"{N_PEOPLE} people {a}", WORDINGS[0][1]])


def items(quick):
    """pilot 10's sentences with their price, W0 rating (8 people), and 4 nearest-priced partners of other values."""
    P10 = ROOT / "pilot10-full-range"
    S = json.load(open(P10 / "sentences10.json"))["sentences"]
    price = {d["text"]: d["u"] for d in json.load(open(P10 / "results/real/prices10.json"))}
    R = json.load(open(P10 / "results/real/ratings10.json"))
    rate = {tuple(k): x for k, x in zip(R["keys"], R["rating"])}
    out = [dict(s, u=price[s["text"]], rating=rate[(i, N_PEOPLE, 0)]) for i, s in enumerate(S)]
    if quick:
        vals = sorted({s["value"] for s in out})[:6]
        out = [s for s in out if s["value"] in vals]
    for s in out:
        cands = sorted([j for j, t in enumerate(out) if t["value"] != s["value"]], key=lambda j: abs(out[j]["u"] - s["u"]))
        s["partners"] = cands[:4]
    return out


def encode(tok, user, marker, text):
    """Token ids of the chat prompt and the indices of the sentence's own tokens (the text after `marker`)."""
    s = p7.chat_text(tok, user)
    e = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    start = s.index(user) + user.index(marker) + len(marker)
    end = start + len(text)
    span = [i for i, (x, y) in enumerate(e["offset_mapping"]) if y > start and x < end and y > x]
    assert span and s[start:end] == text, (user, text)
    return e["input_ids"], span


class Runner:
    def __init__(self, name):
        import torch
        self.torch = torch
        self.tok, self.model, self.dev = load_model({"model": name, "dtype": "bfloat16"})
        self.blocks = self.model.model.layers
        self.n_layers = len(self.blocks)
        self.d = self.model.config.hidden_size
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.AB = letter_set(self.tok, "AB")
        digits = [sorted({e[0] for s in (d, " " + d) if len(e := self.tok.encode(s, add_special_tokens=False)) == 1})
                  for d in "0123456789"]
        self.digits = digits
        flat = sorted({t for g in digits for t in g})
        self.col = {t: j for j, t in enumerate(flat)}
        self.W = self.model.lm_head.weight[flat].float()

    def _batch(self, encs, ch):
        torch = self.torch
        T = max(len(encs[i][0]) for i in ch)
        ids = torch.full((len(ch), T), self.pad, dtype=torch.long)
        att = torch.zeros((len(ch), T), dtype=torch.long)
        spans = []
        for j, i in enumerate(ch):
            n = len(encs[i][0])
            ids[j, T - n:] = torch.tensor(encs[i][0])
            att[j, T - n:] = 1
            spans.append([T - n + k for k in encs[i][1]])
        return ids.to(self.dev), att.to(self.dev), spans

    def read(self, encs, layers, bs=16):
        """Mean over the sentence's tokens and its last token, after each block in `layers` (0 = embeddings)."""
        torch = self.torch
        mean = np.zeros((len(encs), len(layers), self.d), dtype=np.float32)
        last = np.zeros_like(mean)
        for ch in batches(len(encs), bs):
            ids, att, spans = self._batch(encs, ch)
            cap = {}
            hs = []
            for li, L in enumerate(layers):
                mod = self.model.model.embed_tokens if L == 0 else self.blocks[L - 1]
                hs.append(mod.register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(li, o[0] if isinstance(o, tuple) else o)))
            try:
                with torch.no_grad():
                    self.model.model(input_ids=ids, attention_mask=att)
            finally:
                for h in hs:
                    h.remove()
            for li in range(len(layers)):
                H = cap[li].float()
                for j, i in enumerate(ch):
                    mean[i, li] = H[j, spans[j]].mean(0).cpu().numpy()
                    last[i, li] = H[j, spans[j][-1]].cpu().numpy()
        return mean, last

    def _push_hook(self, L, vec, deltas, spans):
        torch = self.torch
        v = torch.tensor(vec, dtype=torch.float32, device=self.dev)

        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            for j, sp in enumerate(spans):
                if deltas[j] != 0:
                    h[j, sp] += (deltas[j] * v).to(h.dtype)
        return self.blocks[L - 1].register_forward_hook(hook)

    def menu_logit(self, encs, push=None, bs=32):
        """logit P(A) (A's letters against B's); push = (layer, unit vector, delta per prompt) adds delta * vector at the
        sentence's tokens after that block."""
        torch = self.torch
        out = np.zeros(len(encs))
        for ch in batches(len(encs), bs):
            ids, att, spans = self._batch(encs, ch)
            h = self._push_hook(push[0], push[1], [push[2][i] for i in ch], spans) if push else None
            try:
                lg = forward(self.model, ids, att, torch).logits[:, -1].float()
            finally:
                if h:
                    h.remove()
            lp = torch.log_softmax(lg, -1)
            out[ch] = (torch.logsumexp(lp[:, self.AB[0]], -1) - torch.logsumexp(lp[:, self.AB[1]], -1)).cpu().numpy()
        return out

    def rating(self, encs, push=None, bs=32):
        """The digits' log-probabilities (float32 from the final hidden state), with an optional push."""
        torch = self.torch
        LP = np.zeros((len(encs), 10))
        for ch in batches(len(encs), bs):
            ids, att, spans = self._batch(encs, ch)
            h = self._push_hook(push[0], push[1], [push[2][i] for i in ch], spans) if push else None
            try:
                with torch.no_grad():
                    hid = self.model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, -1].float()
            finally:
                if h:
                    h.remove()
            lg = hid @ self.W.T
            L = torch.stack([torch.logsumexp(lg[:, [self.col[t] for t in g]], -1) for g in self.digits], 1)
            LP[ch] = torch.log_softmax(L, -1).double().cpu().numpy()
        return LP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    name = "Qwen/Qwen3-0.6B" if a.quick else "Qwen/Qwen3-14B"
    res = HERE / "results" / ("quick" if a.quick else "real")
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S = items(a.quick)
    json.dump(S, open(res / "items11.json", "w"), indent=1)
    M = Runner(name)
    layers = list(range(0, M.n_layers + 1, 4))
    menu_enc = [encode(M.tok, menu_user(s["text"], S[s["partners"][0]]["text"]), f"A) {N_PEOPLE} people ", s["text"]) for s in S]
    rate_enc = [encode(M.tok, rating_user(s["text"]), f"\n{N_PEOPLE} people ", s["text"]) for s in S]
    log(f"{name}: {len(S)} sentences; layers {layers}")

    # ---- read
    Xm, Xm_last = M.read(menu_enc, layers)
    Xr, Xr_last = M.read(rate_enc, layers)
    np.savez_compressed(res / "states11.npz", menu_mean=Xm.astype(np.float16), menu_last=Xm_last.astype(np.float16),
                        rating_mean=Xr.astype(np.float16), rating_last=Xr_last.astype(np.float16), layers=np.array(layers))
    log("read done")

    # ---- probe
    u = np.array([s["u"] for s in S])
    rt = np.array([s["rating"] for s in S])
    vals = sorted({s["value"] for s in S})
    groups = np.array([vals.index(s["value"]) for s in S])
    levels = np.array([["small", "medium", "large"].index(s["level"]) for s in S])
    dis = (rt - rt.mean()) / rt.std() - (u - u.mean()) / u.std()
    table = []
    for li, L in enumerate(layers):
        row = {"layer": L}
        for ctx, X, XL in (("menu", Xm, Xm_last), ("rating", Xr, Xr_last)):
            row[f"{ctx}_mean_price"] = pb.r(pb.lovo_dom(X[:, li], u, groups)[0], u)
            row[f"{ctx}_last_price"] = pb.r(pb.lovo_dom(XL[:, li], u, groups)[0], u)
            row[f"{ctx}_mean_rating"] = pb.r(pb.lovo_dom(X[:, li], rt, groups)[0], rt)
        row["ridge_menu_mean_price"] = pb.r(pb.lovo_ridge(Xm[:, li], u, groups), u)
        row["ridge_rating_mean_rating"] = pb.r(pb.lovo_ridge(Xr[:, li], rt, groups), rt)
        _, other = pb.lovo_dom(Xm[:, li], u, groups, Xr[:, li])
        row["transfer_to_rating_ctx_r_rating"] = pb.r(other, rt)
        row["transfer_partial_price_rating"] = pb.partial_r(u, rt, other)
        row["disagreement_rating_ctx"] = pb.r(pb.lovo_dom(Xr[:, li], dis, groups)[0], dis)
        table.append(row)
    cand = [i for i, L in enumerate(layers) if SELECT[0] <= L <= SELECT[1]]
    li_star = max(cand, key=lambda i: table[i]["menu_mean_price"])
    L_star = layers[li_star]
    pred_star, other_star = pb.lovo_dom(Xm[:, li_star], u, groups, Xr[:, li_star])
    q1 = table[li_star]["menu_mean_price"]
    q2_r, q2_partial = pb.r(other_star, rt), pb.partial_r(u, rt, other_star)
    wl = lambda x: pb.within_level(x, levels)
    within = {"price_rating_r": pb.r(wl(u), wl(rt)), "reading_price_r": pb.r(wl(pred_star), wl(u)),
              "reading_rating_r": pb.r(wl(other_star), wl(rt)), "partial": pb.partial_r(wl(u), wl(rt), wl(other_star))}
    v_price, _, _ = pb.full_dom(Xm[:, li_star], u)
    v_rating, _, _ = pb.full_dom(Xr[:, li_star], rt)
    cos_pr = float(v_price @ v_rating)
    log(f"probe done: layer {L_star}, Q1 r {q1:.2f}, Q2 r {q2_r:.2f}, partial {q2_partial:.2f}")

    # ---- steer
    menu_items = [(i, j) for i, s in enumerate(S) for j in s["partners"]]
    menu_enc_all = [encode(M.tok, menu_user(S[i]["text"], S[j]["text"]), f"A) {N_PEOPLE} people ", S[i]["text"]) for i, j in menu_items]
    base_menu = M.menu_logit(menu_enc_all)
    base_rate = M.rating(rate_enc)
    rng = np.random.default_rng(11)
    steer = {}
    steer_layers = sorted({L for L in layers if 4 <= L <= M.n_layers - 4} | {L_star})
    for L in steer_layers:
        li = layers.index(L)
        v, _, c = pb.full_dom(Xm[:, li], u)
        kinds = [("vibe", v)]
        if L == L_star:
            rv = rng.normal(size=M.d)
            kinds.append(("random", rv / np.linalg.norm(rv)))
        for kind, vec in kinds:
            dl = PUSH / c                                  # residual units for +PUSH price units along v
            mp = M.menu_logit(menu_enc_all, (L, vec, np.full(len(menu_enc_all), dl)))
            mm = M.menu_logit(menu_enc_all, (L, vec, np.full(len(menu_enc_all), -dl)))
            rp = M.rating(rate_enc, (L, vec, np.full(len(rate_enc), dl)))
            rm = M.rating(rate_enc, (L, vec, np.full(len(rate_enc), -dl)))
            er = lambda LP: np.exp(LP) @ np.arange(10)
            steer[f"{kind}@{L}"] = {
                "layer": L, "kind": kind, "delta_residual": float(dl), "c": float(c),
                "menu_transfer": float(np.mean(mp - mm) / (2 * PUSH)),
                "rating_transfer": float(np.mean(er(rp) - er(rm)) / (2 * PUSH * BETA)),
                "rating_points_per_price_unit": float(np.mean(er(rp) - er(rm)) / (2 * PUSH)),
                "score_change_per_price_unit": float(np.mean(log_scale_score(rp) - log_scale_score(rm)) / (2 * PUSH)),
                "menu_transfer_by_level": {lev: float(np.mean((mp - mm)[[levels[i] == k for i, _ in menu_items]]) / (2 * PUSH))
                                           for k, lev in enumerate(("small", "medium", "large"))}}
        log(f"steer layer {L}: menu transfer {steer[f'vibe@{L}']['menu_transfer']:.2f}, rating transfer {steer[f'vibe@{L}']['rating_transfer']:.2f}")
    vs, rs = steer[f"vibe@{L_star}"], steer[f"random@{L_star}"]
    q3 = (vs["menu_transfer"] >= Q3_MIN and vs["rating_transfer"] >= Q3_MIN
          and abs(rs["menu_transfer"]) <= Q3_RANDOM_MAX and abs(rs["rating_transfer"]) <= Q3_RANDOM_MAX)

    # ---- report
    p1, p2 = q1 >= Q1_MIN, (q2_r >= Q2_R_MIN and q2_partial <= Q2_PARTIAL_MAX)
    verdict = ("one vibe direction feeds both" if p1 and p2 and q3 else "readable but not shared" if p1 and not p2
               else "shared but not used" if p1 and p2 else "no readable price on the sentence")
    V = {"model": name, "layer": L_star, "Q1": {"r": q1, "pass": p1}, "Q2": {"r_rating": q2_r, "partial": q2_partial, "raw_r": pb.r(u, rt), "pass": p2},
         "Q3": {"vibe": vs, "random": rs, "pass": q3}, "verdict": verdict, "within_level": within, "cos_price_rating_direction": cos_pr,
         "table": table, "steer": steer, "minutes": (time.time() - t0) / 60,
         "base_menu_logit_abs_median": float(np.median(np.abs(base_menu))),
         "base_rating_mean": float((np.exp(base_rate) @ np.arange(10)).mean())}
    json.dump(V, open(res / "verdicts11.json", "w"), indent=1, default=float)
    readings = [{"value": s["value"], "level": s["level"], "text": s["text"], "u": s["u"], "rating": s["rating"],
                 "reading_menu_ctx": float(pred_star[i]), "projection_rating_ctx": float(other_star[i])} for i, s in enumerate(S)]
    json.dump(readings, open(res / "readings11.json", "w"), indent=1)
    L = [f"# Pilot 11 results ({name})", "",
         f"{len(S)} sentences; the vibe direction (difference of means, price thirds) learned on all but one value and read on that value. "
         f"Layer chosen by Q1 (among {SELECT[0]}-{SELECT[1]}): **{L_star}**.", "",
         "## Pre-registered questions", "",
         f"- **Q1** readable before option B: held-out r = {q1:.2f}: **{'pass' if p1 else 'fail'}** (bar {Q1_MIN})",
         f"- **Q2** the same direction read in the rating prompt: r with the rating = {q2_r:.2f}; price and rating correlate {pb.r(u, rt):.2f}, "
         f"and {q2_partial:.2f} once the reading is accounted for: **{'pass' if p2 else 'fail'}** (bars {Q2_R_MIN} and {Q2_PARTIAL_MAX})",
         f"- **Q3** pushing along it by ±{PUSH:g} price units at layer {L_star}: menu transfer {vs['menu_transfer']:.2f}, rating transfer "
         f"{vs['rating_transfer']:.2f}; a random direction: {rs['menu_transfer']:.2f} and {rs['rating_transfer']:.2f}: "
         f"**{'pass' if q3 else 'fail'}** (bars {Q3_MIN}; random at most {Q3_RANDOM_MAX})", "",
         f"**Verdict: {verdict}.**", "",
         "## Within stakes level (each level's mean removed)", "",
         f"Price and rating r = {within['price_rating_r']:.2f}; the reading against the price {within['reading_price_r']:.2f}; "
         f"the rating-prompt projection against the rating {within['reading_rating_r']:.2f}; price and rating once it is accounted for {within['partial']:.2f}.",
         f"Cosine between the price direction (menu prompt) and a rating direction (rating prompt) at layer {L_star}: {cos_pr:.2f}.", "",
         "## Every layer (held-out r; vibe direction unless marked ridge)", "",
         "| layer | menu, mean: price | menu, last: price | rating ctx, mean: price | rating ctx: rating | ridge menu: price | ridge rating ctx: rating | menu direction read in rating ctx: rating | partial price-rating | disagreement |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        L.append(f"| {t['layer']} | {t['menu_mean_price']:.2f} | {t['menu_last_price']:.2f} | {t['rating_mean_price']:.2f} | {t['rating_mean_rating']:.2f} | "
                 f"{t['ridge_menu_mean_price']:.2f} | {t['ridge_rating_mean_rating']:.2f} | {t['transfer_to_rating_ctx_r_rating']:.2f} | "
                 f"{t['transfer_partial_price_rating']:.2f} | {t['disagreement_rating_ctx']:.2f} |")
    L += ["", f"## Pushing along the direction (±{PUSH:g} price units)", "",
          "| layer | kind | menu transfer (small / medium / large) | rating transfer | rating points per price unit | log-scale score per price unit |", "|---|---|---|---|---|---|"]
    for k, s in sorted(steer.items(), key=lambda kv: (kv[1]["layer"], kv[1]["kind"])):
        b = s["menu_transfer_by_level"]
        L.append(f"| {s['layer']} | {s['kind']} | {s['menu_transfer']:.2f} ({b['small']:.2f} / {b['medium']:.2f} / {b['large']:.2f}) | "
                 f"{s['rating_transfer']:.2f} | {s['rating_points_per_price_unit']:.3f} | {s['score_change_per_price_unit']:.3f} |")
    L += ["", f"Base menu prompts: median |logit P(A)| {V['base_menu_logit_abs_median']:.2f}. Run time {V['minutes']:.1f} min."]
    (res / "report.md").write_text("\n".join(L) + "\n")
    log(f"report written; {V['minutes']:.1f} min")


if __name__ == "__main__":
    main()
