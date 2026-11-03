"""Pilot 12: one vibe, or shared ingredients with different recipes? (DESIGN.md)

  python pilot12.py            Qwen3-14B: results/real/
  python pilot12.py --quick    Qwen3-0.6B on the first 6 values (24 sentences): results/quick/ (a pipeline test)

Reuses pilot 11's model runner (reading the states, pushing along a direction) and its prompts.
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
          "pilot11-vibe-direction", "pilot12-recipes"):
    sys.path.insert(0, str(ROOT / p))
import pilot11 as p11  # noqa: E402
import probe11 as pb  # noqa: E402
import probe12 as q  # noqa: E402
from common import log  # noqa: E402

Q1_MIN, Q1_P, N_PERM, N_BOOT = 0.3, 0.05, 200, 2000
Q1_LAYERS = (8, 32)
PUSH_LAYER, PUSH = 20, 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    name = "Qwen/Qwen3-0.6B" if a.quick else "Qwen/Qwen3-14B"
    res = HERE / "results" / ("quick" if a.quick else "real")
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S = p11.items(a.quick)
    P10 = ROOT / "pilot10-full-range"
    S10 = json.load(open(P10 / "sentences10.json"))["sentences"]
    idx10 = {s["text"]: i for i, s in enumerate(S10)}
    R10 = json.load(open(P10 / "results/real/ratings10.json"))
    rate = {tuple(k): x for k, x in zip(R10["keys"], R10["rating"])}
    u = np.array([s["u"] for s in S])
    Rbar = np.array([np.mean([rate[(idx10[s["text"]], 8, w)] for w in (0, 1, 2)]) for s in S])
    W0 = np.array([s["rating"] for s in S])
    d = q.cubic_residual(u, Rbar)
    vals = sorted({s["value"] for s in S})
    groups = np.array([vals.index(s["value"]) for s in S])

    M = p11.Runner(name)
    layers = list(range(0, M.n_layers + 1, 4))
    menu_enc = [p11.encode(M.tok, p11.menu_user(s["text"], S[s["partners"][0]]["text"]), f"A) {p11.N_PEOPLE} people ", s["text"]) for s in S]
    rate_enc = [p11.encode(M.tok, p11.rating_user(s["text"]), f"\n{p11.N_PEOPLE} people ", s["text"]) for s in S]
    log(f"{name}: {len(S)} sentences; layers {layers}")
    Xm, _ = M.read(menu_enc, layers)
    Xr, _ = M.read(rate_enc, layers)
    np.savez_compressed(res / "states12.npz", menu_mean=Xm.astype(np.float16), rating_mean=Xr.astype(np.float16),
                        layers=np.array(layers), u=u, Rbar=Rbar, W0=W0, d=d)
    log("read done")

    # ---- Q1: is the disagreement readable from the sentence (menu context)?
    D = M.d
    q1_li = [i for i, L in enumerate(layers) if Q1_LAYERS[0] <= L <= Q1_LAYERS[1]]
    models_m = [q.RidgeLOVO(q.kernel(Xm[:, i]), groups, D) for i in q1_li]
    rs_m, bi, q1_r = q.best_layer_r(models_m, d)
    q1_layer = layers[q1_li[bi]]
    q1_p, null = q.permutation_p(models_m, d, N_PERM, seed=12)
    q1_pass = q1_r >= Q1_MIN and q1_p < Q1_P
    d_hat = models_m[bi].predict(d)
    models_r = [q.RidgeLOVO(q.kernel(Xr[:, i]), groups, D) for i in q1_li]
    rs_r = [pb.r(m.predict(d), d) for m in models_r]
    rs_dom = [pb.r(pb.lovo_dom(Xm[:, i], d, groups)[0], d) for i in q1_li]
    log(f"Q1: layer {q1_layer}, r {q1_r:.2f}, permutation p {q1_p:.3f}")

    # ---- a small set of directions (PLS), menu context at Q1's layer
    li_q1 = layers.index(q1_layer)
    pls = {}
    for k in (1, 2, 3, 5):
        P = q.pls2_lovo(Xm[:, li_q1], np.column_stack([u, Rbar]), groups, k)
        pls[k] = {"price_r": pb.r(P[:, 0], u), "rating_r": pb.r(P[:, 1], Rbar), "partial": q.partial_r_multi(u, Rbar, P)}

    # ---- Q2: pushes at layer 20
    li = layers.index(PUSH_LAYER) if PUSH_LAYER in layers else len(layers) // 2
    L = layers[li]
    v_price, _, c = pb.full_dom(Xm[:, li], u)
    dirs = {"price": v_price, "rating": pb.dom_direction(Xm[:, li], Rbar), "disagreement": pb.dom_direction(Xm[:, li], d)}
    rv = np.random.default_rng(12).normal(size=D)
    dirs["random"] = rv / np.linalg.norm(rv)
    dl = PUSH / c
    menu_items = [(i, j) for i, s in enumerate(S) for j in s["partners"]]
    menu_all = [p11.encode(M.tok, p11.menu_user(S[i]["text"], S[j]["text"]), f"A) {p11.N_PEOPLE} people ", S[i]["text"]) for i, j in menu_items]
    owner = np.array([i for i, _ in menu_items])
    er = lambda LP: np.exp(LP) @ np.arange(10)
    pushes = {}
    for kname, vec in dirs.items():
        mp = M.menu_logit(menu_all, (L, vec, np.full(len(menu_all), dl)))
        mm = M.menu_logit(menu_all, (L, vec, np.full(len(menu_all), -dl)))
        rp = M.rating(rate_enc, (L, vec, np.full(len(rate_enc), dl)))
        rm = M.rating(rate_enc, (L, vec, np.full(len(rate_enc), -dl)))
        Cs = np.array([np.mean((mp - mm)[owner == i]) for i in range(len(S))]) / (2 * dl)      # logit per residual unit
        Rs = (er(rp) - er(rm)) / (2 * dl)                                                    # rating points per residual unit
        pushes[kname] = {"C": Cs, "R": Rs}
        log(f"push {kname}: choice {Cs.mean() * dl:.3f} logit and rating {Rs.mean() * dl:.3f} points per {PUSH:g}-price-unit push")
    ex = {k: q.bootstrap_excess(pushes["price"]["C"], pushes["price"]["R"], pushes[k]["C"], pushes[k]["R"], N_BOOT, seed=12)
          for k in ("disagreement", "rating", "random")}
    q2_est, q2_ci = ex["disagreement"]
    q2_pass = q2_ci[0] > 0
    verdict = ("different recipes" if q1_pass and q2_pass else "one vibe" if not q1_pass and not q2_pass else "mixed")

    # ---- report
    s_ratio = pushes["price"]["R"].mean() / pushes["price"]["C"].mean()
    V = {"model": name, "Q1": {"r": q1_r, "layer": q1_layer, "p": q1_p, "pass": q1_pass, "r_by_layer_menu": dict(zip([layers[i] for i in q1_li], rs_m)),
                               "r_by_layer_rating_ctx": dict(zip([layers[i] for i in q1_li], rs_r)), "r_by_layer_dom": dict(zip([layers[i] for i in q1_li], rs_dom)),
                               "null_95": float(np.quantile(null, 0.95))},
         "Q2": {"excess": q2_est, "ci95": q2_ci, "pass": q2_pass, "rating_per_logit_price_direction": float(s_ratio),
                "push_layer": L, "delta_residual": float(dl)},
         "excess_other": {k: {"excess": v[0], "ci95": v[1]} for k, v in ex.items()},
         "pushes": {k: {"choice_logit_per_2_price_units": float(v["C"].mean() * dl), "rating_points_per_2_price_units": float(v["R"].mean() * dl)}
                    for k, v in pushes.items()},
         "pls": pls, "verdict": verdict, "raw_r_price_rating": pb.r(u, Rbar), "minutes": (time.time() - t0) / 60}
    json.dump(V, open(res / "verdicts12.json", "w"), indent=1, default=float)
    by_val = {v: float(np.mean(d_hat[groups == i])) for i, v in enumerate(vals)}
    true_val = {v: float(np.mean(d[groups == i])) for i, v in enumerate(vals)}
    order = sorted(vals, key=lambda v: by_val[v])
    json.dump([{"value": s["value"], "level": s["level"], "text": s["text"], "u": float(u[i]), "Rbar": float(Rbar[i]), "d": float(d[i]),
                "d_heldout": float(d_hat[i])} for i, s in enumerate(S)], open(res / "disagreement12.json", "w"), indent=1)
    Lr = [f"# Pilot 12 results ({name})", "",
          f"{len(S)} sentences. Rating = mean of W0-W2 at 8 people; it correlates {pb.r(u, Rbar):.2f} with the price. "
          f"Disagreement = the rating minus a cubic in the price (SD {d.std():.2f} rating points).", "",
          "## Pre-registered questions", "",
          f"- **Q1** the disagreement read from the sentence as option A (ridge, held out by value): r = {q1_r:.2f} at layer {q1_layer}; "
          f"permutation p = {q1_p:.3f} (95th percentile of the null {np.quantile(null, 0.95):.2f}): **{'pass' if q1_pass else 'fail'}** "
          f"(bars r {Q1_MIN}, p {Q1_P})",
          f"- **Q2** pushing along the disagreement direction at layer {L}: excess = {q2_est:+.4f} rating points per residual unit "
          f"[{q2_ci[0]:+.4f}, {q2_ci[1]:+.4f}]: **{'pass' if q2_pass else 'fail'}** (the interval must lie above 0)", "",
          f"**Verdict: {verdict}.**", "",
          "## Pushes (each by the residual distance of 2 price units along the price direction)", "",
          "| direction | choice: change in logit P(A) | rating: change in points | excess over the price direction's ratio (95% interval) |", "|---|---|---|---|"]
    for k, v in pushes.items():
        e = f"{ex[k][0]:+.4f} [{ex[k][1][0]:+.4f}, {ex[k][1][1]:+.4f}]" if k in ex else "(the reference)"
        Lr.append(f"| {k} | {v['C'].mean() * dl:+.3f} | {v['R'].mean() * dl:+.3f} | {e} |")
    Lr += ["", f"The price direction moves the rating {s_ratio * 1:.3f} points per logit of choice. Excess is in rating points per residual unit; "
           f"multiply by {dl:.2f} for points per push.", "",
           "## Q1 by layer (held-out r of the disagreement)", "",
           "| layer | ridge, menu context | ridge, rating context | difference of means, menu context |", "|---|---|---|---|"]
    for i, Li in enumerate([layers[j] for j in q1_li]):
        Lr.append(f"| {Li} | {rs_m[i]:.2f} | {rs_r[i]:.2f} | {rs_dom[i]:.2f} |")
    Lr += ["", f"## A small set of directions (PLS, menu context, layer {q1_layer}, held out by value)", "",
           f"Price and rating correlate {pb.r(u, Rbar):.2f}. With noisy held-out readouts this partial correlation cannot reach 0 even if the overlap lives in these directions.", "",
           "| components | price r | rating r | partial r of price and rating given both readouts |", "|---|---|---|---|"]
    for k, v in pls.items():
        Lr.append(f"| {k} | {v['price_r']:.2f} | {v['rating_r']:.2f} | {v['partial']:.2f} |")
    Lr += ["", "## Where the held-out disagreement readout puts each value (mean of its 4 sentences; actual in brackets)", "",
           "Rated below its price: " + ", ".join(f"{v} {by_val[v]:+.2f} ({true_val[v]:+.2f})" for v in order[:5]) + ".",
           "Rated above its price: " + ", ".join(f"{v} {by_val[v]:+.2f} ({true_val[v]:+.2f})" for v in order[-5:][::-1]) + ".", "",
           f"Run time {V['minutes']:.1f} min."]
    (res / "report.md").write_text("\n".join(Lr) + "\n")
    log(f"report written; {V['minutes']:.1f} min")


if __name__ == "__main__":
    main()
