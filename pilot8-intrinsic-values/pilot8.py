"""Pilot 8: trade-offs between intrinsic values in Qwen3-14B (DESIGN.md).

  python pilot8.py                 all stages
  python pilot8.py --quick         Qwen3-0.6B and a subset (a pipeline test)

Stages:
  behave   trade-offs: P(A) for every two-option prompt; each value's worth (the price list); one scale,
           transitivity (Q1, Q3); three-option menus (menu invariance, Q3); stated ratings (Q4a)
  source   the worth direction on option A's own words, trained on the 12 training values; does it order
           the 12 held-out values by worth (Q2a)
  compare  the comparison direction after option B, trained on trade-offs among the training values;
           does it carry trade-offs among the held-out values (Q2b)
  rating   the trade-off worth direction in the rating prompts: does it order the values there (Q4b),
           and does swapping it change the stated rating (Q4c)
  report   verdicts and report.md
Reuses pilot 7's sites, patching and distributed alignment search.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
P5, P7 = HERE.parent / "pilot5-value-circuit", HERE.parent / "pilot7-value-route"
sys.path.insert(0, str(P7))
sys.path.insert(0, str(P5))
import pilot7 as p7  # noqa: E402
from common import Hooks, batches, deep_update, forward, layer_output, log, save_json  # noqa: E402


# --------------------------------------------------------------------------
# Settings and data
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stages", default="behave,source,compare,rating,report")
    return ap.parse_args()


def load_cfg(args):
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.quick:
        q = cfg.get("quick", {})
        cfg = deep_update(cfg, {k: v for k, v in q.items() if k != "models"})
        cfg["models"] = q["models"]
    cfg["quick_mode"] = bool(args.quick)
    res = Path(cfg["results_dir"])
    cfg["res"] = str(res if res.is_absolute() else HERE / res)
    return cfg


def jl(name):
    return [json.loads(l) for l in open(HERE / name, encoding="utf-8")]


class Data:
    def __init__(self, cfg):
        lim = cfg.get("limit") or {}
        rng = np.random.default_rng(0)
        sub = lambda xs, k: xs if not k or len(xs) <= k else [xs[i] for i in sorted(rng.choice(len(xs), size=k, replace=False))]
        pairs = jl("pairs.jsonl")
        self.sets = {s: sub([p for p in pairs if p["set"] == s], lim.get(s)) for s in ("das_train", "das_test")}
        self.rows = sub(jl("behaviour.jsonl"), lim.get("behaviour"))
        menus = jl("menus.jsonl")
        if lim.get("menus"):
            menus = menus[: lim["menus"]]
        self.menus = menus
        self.ratings = jl("ratings.jsonl")
        self.rating_pairs = sub(jl("rating_pairs.jsonl"), lim.get("rating_pairs"))
        V = json.load(open(HERE / "values.json"))
        self.names = [v["name"] for v in V["values"]]
        self.phrasings = {v["name"]: v["phrasings"] for v in V["values"]}
        self.train, self.test = V["train"], V["test"]


def letter_set(tok, letters):
    out = []
    for L in letters:
        ids = {e[0] for s in (L, " " + L) if len(e := tok.encode(s, add_special_tokens=False)) == 1}
        out.append(sorted(ids))
    return out


def probs(o, groups, torch):
    pr = torch.softmax(o.logits[:, -1].float(), -1)
    return torch.stack([pr[:, g].sum(-1) for g in groups], 1).cpu().numpy()


# --------------------------------------------------------------------------
# Fitting the worth
# --------------------------------------------------------------------------
def fit_worth(ia, ib, dlog, p, n_out, init=None):
    """P(A) = sigmoid(u_A - u_B + b dlog + c), u_0 = 0; maximum likelihood on soft labels p."""
    from scipy.optimize import minimize
    p = np.clip(p, 1e-6, 1 - 1e-6)

    def f(x):
        u = np.concatenate([[0.0], x[:n_out - 1]])
        z = u[ia] - u[ib] + x[n_out - 1] * dlog + x[n_out]
        q = np.clip(1 / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
        g = q - p
        gu = np.bincount(ia, g, n_out) - np.bincount(ib, g, n_out)
        return -np.sum(p * np.log(q) + (1 - p) * np.log(1 - q)), np.concatenate([gu[1:], [np.sum(g * dlog)], [np.sum(g)]])
    r = minimize(f, np.zeros(n_out + 1) if init is None else init, jac=True, method="L-BFGS-B")
    return np.concatenate([[0.0], r.x[:n_out - 1]]), float(r.x[n_out - 1]), float(r.x[n_out]), r.x


def logodds(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def cyclic_triads(names, pref):
    """pref[(x, y)] = mean probability that x is chosen over y. Counts 3-cycles of the majority relation."""
    beats = lambda x, y: pref[(x, y)] > 0.5
    cyc, strong = 0, 0
    for x, y, z in itertools.combinations(names, 3):
        for a, b, c in ((x, y, z), (x, z, y)):
            if beats(a, b) and beats(b, c) and beats(c, a):
                cyc += 1
                strong += min(pref[(a, b)], pref[(b, c)], pref[(c, a)]) > 0.6
    return cyc, strong


# --------------------------------------------------------------------------
# Stage: behave (Q1, Q3, Q4a)
# --------------------------------------------------------------------------
def stage_behave(M, D, cfg):
    import torch
    tok = M.tok
    LAB, LAB3 = letter_set(tok, "AB"), letter_set(tok, "ABC")
    DIG = letter_set(tok, "0123456789")

    def run(users, groups, batch_fn=M.batch):
        out = np.zeros((len(users), len(groups)))
        with torch.no_grad():
            for ch in batches(len(users), int(cfg["batch_size"])):
                ids, att, _ = batch_fn([users[i] for i in ch])
                out[ch] = probs(forward(M.model, ids, att, torch), groups, torch)
        return out
    rows = D.rows
    P = run([r["user"] for r in rows], LAB)
    pa = P[:, 0] / P.sum(1)
    idx = {k: i for i, k in enumerate(D.names)}
    ia, ib = np.array([idx[r["a"]] for r in rows]), np.array([idx[r["b"]] for r in rows])
    dlog = np.array([math.log(r["na"] / r["nb"]) for r in rows])
    n = len(D.names)
    u, b, c, x = fit_worth(ia, ib, dlog, pa, n)
    # sentence-level worth (phrasing reliability)
    sid = {(v, p): i for i, (v, p) in enumerate(itertools.product(D.names, (0, 1)))}
    sa = np.array([sid[(r["a"], r["pa"])] for r in rows])
    sb = np.array([sid[(r["b"], r["pb"])] for r in rows])
    us, _, _, _ = fit_worth(sa, sb, dlog, pa, len(sid))
    u_p0 = np.array([us[sid[(v, 0)]] for v in D.names])
    u_p1 = np.array([us[sid[(v, 1)]] for v in D.names])
    # Q3 (i) one scale: fitted on wording 0, predicting wording 1
    w0 = np.array([r["wording"] == 0 for r in rows])
    u0, b0, c0, _ = fit_worth(ia[w0], ib[w0], dlog[w0], pa[w0], n)
    z1 = u0[ia[~w0]] - u0[ib[~w0]] + b0 * dlog[~w0] + c0
    one_r = float(np.corrcoef(z1, logodds(pa[~w0]))[0, 1])
    one_acc = float(((z1 > 0) == (pa[~w0] > 0.5)).mean())
    # Q3 (ii) transitivity at equal numbers
    acc = {}
    for r, p in zip(rows, pa):
        if r["na"] != r["nb"]:
            continue
        acc.setdefault((r["a"], r["b"]), []).append(p)
        acc.setdefault((r["b"], r["a"]), []).append(1 - p)
    pref = {k: float(np.mean(v)) for k, v in acc.items()}
    for xx, yy in itertools.permutations(D.names, 2):
        pref.setdefault((xx, yy), 0.5)
    cyc, strong = cyclic_triads(D.names, pref)
    n_tri = math.comb(n, 3)
    # Q3 (iii) menu invariance
    mu = [m["user"] for m in D.menus]
    three = [i for i, m in enumerate(D.menus) if m["kind"] == "three"]
    twos = [i for i, m in enumerate(D.menus) if m["kind"] == "two"]
    P3 = run([mu[i] for i in three], LAB3)
    P2 = run([mu[i] for i in twos], LAB)
    shift = np.log(P3[:, 0] / P3[:, 1]) - np.log(P2[:, 0] / P2[:, 1])
    # Q4a stated ratings
    PR = run([r["user"] for r in D.ratings], DIG, RatingEnc(M).batch)
    er = (PR / PR.sum(1, keepdims=True)) @ np.arange(10)
    rate = {v: float(np.mean([e for e, r in zip(er, D.ratings) if r["value"] == v])) for v in D.names}
    rng = np.random.default_rng(int(cfg["fit"]["seed"]))
    boots = []
    for _ in range(int(cfg["fit"]["bootstrap"])):
        s = rng.integers(0, len(rows), len(rows))
        ub, bb, _, _ = fit_worth(ia[s], ib[s], dlog[s], pa[s], n, init=x)
        boots.append(np.concatenate([ub - ub.mean(), [bb]]))
    boots = np.array(boots)
    uc = u - u.mean()
    out = {"n_prompts": len(rows), "b": b, "b_ci95": [float(np.quantile(boots[:, -1], q)) for q in (0.025, 0.975)],
           "bias_towards_A": c, "fit_corr_logodds": float(np.corrcoef(u[ia] - u[ib] + b * dlog + c, logodds(pa))[0, 1]),
           "worth": {v: {"u": float(uc[i]), "ci95": [float(np.quantile(boots[:, i], q)) for q in (0.025, 0.975)],
                         "people_vs_mean": float(math.exp(uc[i] / b)) if b > 0 else float("nan")} for i, v in enumerate(D.names)},
           "phrasing_reliability_r": float(np.corrcoef(u_p0, u_p1)[0, 1]),
           "one_scale": {"heldout_wording_r": one_r, "heldout_wording_same_side": one_acc},
           "transitivity": {"cyclic_triads": cyc, "strong_cyclic_triads": strong, "triads": n_tri, "share": cyc / n_tri},
           "menu": {"n": len(three), "median_abs_log_shift": float(np.median(np.abs(shift))),
                    "median_factor": float(math.exp(np.median(np.abs(shift)))), "mean_log_shift": float(shift.mean()),
                    "share_over_factor_1_5": float((np.abs(shift) > math.log(1.5)).mean())},
           "ratings": rate, "ratings_vs_worth_r": float(np.corrcoef([rate[v] for v in D.names], uc)[0, 1]),
           "ratings_vs_worth_spearman": float(np.corrcoef(np.argsort(np.argsort([rate[v] for v in D.names])), np.argsort(np.argsort(uc)))[0, 1])}
    np.save(M.res / "behaviour_pA.npy", pa.astype(np.float32))
    save_json(out, M.res / "behave.json")
    log(f"  behave: {len(rows)} trade-offs, {len(three)} menus, {len(D.ratings)} ratings")


# --------------------------------------------------------------------------
# Stage: source (Q2a)
# --------------------------------------------------------------------------
def positions_tradeoff(M, D, site, L, w):
    import torch
    sys.path.insert(0, str(HERE))
    from make_prompts import two
    users, own = [], []
    for v in range(len(D.names)):
        other = 0 if v != 0 else 1
        for p in (0, 1):
            for wd in (0, 1):
                users.append(two(v, p, 4, other, 0, 4, wd))
                own.append(D.names[v])
    return _project(M, users, own, site, L, w, M.batch)


def _project(M, users, own, site, L, w, batch_fn):
    import torch
    vals = np.zeros(len(users))
    wt = torch.tensor(w, dtype=torch.float32, device=M.dev)
    with torch.no_grad():
        for ch in batches(len(users), int(M.cfg["batch_size"])):
            cap = {}
            with Hooks() as hk:
                hk.fwd(M.layers[L - 1], lambda m, i, o: cap.__setitem__("h", layer_output(o)))
                ids, att, es = batch_fn([users[i] for i in ch])
                forward(M.model, ids, att, torch)
            T = ids.shape[1]
            for j, i in enumerate(ch):
                kk = torch.tensor(es[j]["sites"][site], device=M.dev)
                vals[i] = float((cap["h"][j, T + kk].float() @ wt).mean())
    return {v: float(np.mean([x for x, o in zip(vals, own) if o == v])) for v in dict.fromkeys(own)}


def stage_source(M, D, cfg):
    beh = json.load(open(M.res / "behave.json"))
    u = {v: beh["worth"][v]["u"] for v in D.names}
    ps = M.pairs["das_train"][: int(cfg["das"]["sweep_pairs"])]
    table, best = {}, None
    for site in cfg["sites"]["a_sites"]:
        for L in [L for L in M.map_layers if L <= int(cfg["das"]["a_layer_max"])]:
            Dd = p7.prep_site(M, ps, site, L)
            t = p7.through_zero(Dd["y_s"] - Dd["y_r"], Dd["y_d"] - Dd["y_r"])["value"]
            table[f"{site}@{L}"] = t
            if best is None or t > best[2] + 1e-12:
                best = (site, L, t)
    site, L, t = best
    log("  source: site chosen by the rule")
    W, hist = p7.das_train(M, p7.prep_site(M, M.pairs["das_train"], site, L), site, L, int(cfg["das"]["seed"]))
    ev = p7.das_eval(M, p7.prep_site(M, M.pairs["das_test"], site, L), site, L, W)
    ev.pop("coord_change")
    w = W[0].cpu().numpy()
    pos = positions_tradeoff(M, D, site, L, w)
    if np.corrcoef([pos[v] for v in D.train], [u[v] for v in D.train])[0, 1] < 0:
        w, pos = -w, {k: -v for k, v in pos.items()}
    out = {"site": site, "layer": L, "site_swap_share": t, "sweep": table, "train_loss_by_epoch": hist, "heldout_swap": ev,
           "positions": pos, "q2a_corr_heldout": float(np.corrcoef([pos[v] for v in D.test], [u[v] for v in D.test])[0, 1]),
           "corr_train": float(np.corrcoef([pos[v] for v in D.train], [u[v] for v in D.train])[0, 1])}
    np.save(M.res / "worth_direction.npy", w.astype(np.float32))
    save_json(out, M.res / "source.json")
    log("  source done")


# --------------------------------------------------------------------------
# Stage: compare (Q2b)
# --------------------------------------------------------------------------
def stage_compare(M, D, cfg):
    ps = M.pairs["das_train"][: int(cfg["das"]["sweep_pairs"])]
    sweep, arrival = {}, None
    for L in [L for L in M.map_layers if L >= int(cfg["das"]["arrival_from"])]:
        Dd = p7.prep_site(M, ps, "last", L)
        t = p7.through_zero(Dd["y_s"] - Dd["y_r"], Dd["y_d"] - Dd["y_r"])["value"]
        sweep[str(L)] = t
        if arrival is None and t >= cfg["prereg"]["arrival_min"]:
            arrival = L
    arrival = arrival or M.map_layers[-1]
    Lc = max([L for L in M.map_layers if L <= arrival - 4] or [M.map_layers[0]])
    site = cfg["sites"]["compare_site"]
    log("  compare: site chosen by the rule")
    W, hist = p7.das_train(M, p7.prep_site(M, M.pairs["das_train"], site, Lc), site, Lc, int(cfg["das"]["seed"]) + 1)
    Dte = p7.prep_site(M, M.pairs["das_test"], site, Lc)
    ev = p7.das_eval(M, Dte, site, Lc, W)
    ev.pop("coord_change")
    out = {"arrival_sweep_last": sweep, "arrival_layer": arrival, "site": site, "layer": Lc, "train_loss_by_epoch": hist,
           "heldout": ev, "q2b_share_heldout": ev["of_site_swap"]["value"]}
    np.save(M.res / "comparison_direction.npy", W[0].cpu().numpy().astype(np.float32))
    save_json(out, M.res / "compare.json")
    log("  compare done")


# --------------------------------------------------------------------------
# Stage: rating (Q4b, Q4c)
# --------------------------------------------------------------------------
class RatingEnc:
    """Rating prompts: the outcome line's last token (R_end) and the line break after it (R_nl)."""
    def __init__(self, M):
        self.M, self.cache = M, {}

    def enc(self, user):
        if user in self.cache:
            return self.cache[user]
        tok = self.M.tok
        s = p7.chat_text(tok, user)
        e = tok(s, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = e["input_ids"], e["offset_mapping"]
        lines = user.split("\n")
        u0 = s.index(user)
        start = u0 + len(lines[0]) + 1 + len(lines[1]) + 1
        end = start + len(lines[2])
        rstart = end + 1
        inline = [i for i, (x, y) in enumerate(offs) if y > start and x < end and y > x]
        r_end = max(inline)
        reply_first = min(i for i, (x, y) in enumerate(offs) if y > rstart and y > x)
        n = len(ids)
        self.cache[user] = {"ids": ids, "a_end": r_end,
                            "sites": {"R_end": [r_end - n], "R_nl": [i - n for i in range(r_end + 1, reply_first)]}}
        return self.cache[user]

    def batch(self, users):
        import torch
        es = [self.enc(x) for x in users]
        T = max(len(x["ids"]) for x in es)
        ids = torch.full((len(es), T), self.M.pad, dtype=torch.long)
        att = torch.zeros((len(es), T), dtype=torch.long)
        for j, x in enumerate(es):
            ids[j, T - len(x["ids"]):] = torch.tensor(x["ids"])
            att[j, T - len(x["ids"]):] = 1
        return ids.to(self.M.dev), att.to(self.M.dev), es


def stage_rating(M, D, cfg):
    import torch
    beh = json.load(open(M.res / "behave.json"))
    src = json.load(open(M.res / "source.json"))
    u = {v: beh["worth"][v]["u"] for v in D.names}
    w = np.load(M.res / "worth_direction.npy")
    L = int(src["layer"])
    rsite = {"A_end": "R_end", "A_nl": "R_nl"}[src["site"]]
    R = RatingEnc(M)
    pos = _project(M, [r["user"] for r in D.ratings], [r["value"] for r in D.ratings], rsite, L, w, R.batch)
    rate = beh["ratings"]
    DIG = letter_set(M.tok, "0123456789")
    rp = [p for p in D.rating_pairs if R.enc(p["user"])["ids"][R.enc(p["user"])["a_end"] + 1:] ==
          R.enc(p["donor_user"])["ids"][R.enc(p["donor_user"])["a_end"] + 1:]]
    Wt = torch.tensor(w[None], dtype=torch.float32, device=M.dev)

    def er(o):
        pr = probs(o, DIG, torch)
        return (pr / pr.sum(1, keepdims=True)) @ np.arange(10)
    y_r, y_d, y_s, y_w = (np.zeros(len(rp)) for _ in range(4))
    with torch.no_grad():
        for ch in batches(len(rp), int(cfg["batch_size"])):
            cap = {}
            with Hooks() as hk:
                hk.fwd(M.layers[L - 1], lambda m, i, o: cap.__setitem__("d", layer_output(o)))
                idd, attd, _ = R.batch([rp[i]["donor_user"] for i in ch])
                y_d[ch] = er(forward(M.model, idd, attd, torch))
            ids, att, es = R.batch([rp[i]["user"] for i in ch])
            y_r[ch] = er(forward(M.model, ids, att, torch))
            rows, ks = M.site_index(es, rsite)
            srcs = cap["d"][rows, idd.shape[1] + ks]
            for target, Wx in ((y_s, None), (y_w, Wt)):
                with Hooks() as hk:
                    hk.fwd(M.layers[L - 1], p7.patch_hook(rows, ks, srcs, Wx))
                    target[ch] = er(forward(M.model, ids, att, torch))
    tz = p7.through_zero(y_w - y_r, y_s - y_r)
    out = {"site": rsite, "layer": L, "positions": pos,
           "q4b_corr_heldout": float(np.corrcoef([pos[v] for v in D.test], [u[v] for v in D.test])[0, 1]),
           "corr_all_worth": float(np.corrcoef([pos[v] for v in D.names], [u[v] for v in D.names])[0, 1]),
           "corr_all_ratings": float(np.corrcoef([pos[v] for v in D.names], [rate[v] for v in D.names])[0, 1]),
           "n_pairs": len(rp), "site_swap_of_donor": p7.through_zero(y_s - y_r, y_d - y_r),
           "mean_abs_rating_change_site": float(np.abs(y_s - y_r).mean()), "mean_abs_rating_diff": float(np.abs(y_d - y_r).mean()),
           "q4c_share": tz}
    save_json(out, M.res / "rating.json")
    log("  rating done")


# --------------------------------------------------------------------------
# Stage: report
# --------------------------------------------------------------------------
def stage_report(cfg, name, D):
    R = Path(cfg["res"]) / name
    pr = cfg["prereg"]
    beh = json.load(open(R / "behave.json"))
    V, L_ = {}, [f"# Pilot 8 results ({cfg['models'][name]['path']})", ""]
    W = beh["worth"]
    order = sorted(W, key=lambda v: -W[v]["u"])
    L_ += ["## Q1. The price list: how the model trades the values off", "",
           f"{beh['n_prompts']} two-option trade-offs; P(A) = σ(u_A − u_B + b·log(n_A/n_B) + c), b = {beh['b']:.2f} "
           f"[{beh['b_ci95'][0]:.2f}, {beh['b_ci95'][1]:.2f}] per log person, bias towards A {beh['bias_towards_A']:+.2f}; the fit "
           f"orders the answers with r = {beh['fit_corr_logodds']:.2f}. The two phrasings agree with r = {beh['phrasing_reliability_r']:.2f}.", "",
           "| rank | value | worth u (95%) | one person counts as … people of an average value | stated rating (0-9) |", "|---|---|---|---|---|"]
    for k, v in enumerate(order, 1):
        L_.append(f"| {k} | {v}{' (held out)' if v in D.test else ''} | {W[v]['u']:+.2f} [{W[v]['ci95'][0]:+.2f}, {W[v]['ci95'][1]:+.2f}] | "
                  f"{W[v]['people_vs_mean']:.2f} | {beh['ratings'][v]:.2f} |")
    o, t, m = beh["one_scale"], beh["transitivity"], beh["menu"]
    q3 = {"one_scale": o["heldout_wording_r"] >= pr["q3_onescale_r"], "transitive": t["share"] <= pr["q3_cyclic_max"],
          "menu_invariant": m["median_factor"] <= pr["q3_menu_factor"]}
    V["Q3"] = {**q3, "verdict": "a consistent menu" if all(q3.values()) else "not consistent: " + ", ".join(k for k, ok in q3.items() if not ok)}
    L_ += ["", "## Q3. Is the menu consistent?", "",
           f"- One scale: prices fitted on wording 1 predict wording 2 with r = {o['heldout_wording_r']:.2f} "
           f"({o['heldout_wording_same_side']:.0%} on the same side): **{'pass' if q3['one_scale'] else 'fail'}**",
           f"- Transitivity: {t['cyclic_triads']} of {t['triads']} triads cyclic ({t['share']:.1%}; {t['strong_cyclic_triads']} with every margin above 0.6): "
           f"**{'pass' if q3['transitive'] else 'fail'}**",
           f"- Menu invariance: adding a third option changes the A:B odds by a median factor {m['median_factor']:.2f} "
           f"({m['share_over_factor_1_5']:.0%} of menus by more than 1.5; mean log shift {m['mean_log_shift']:+.2f}): "
           f"**{'pass' if q3['menu_invariant'] else 'fail'}**", "", f"**Q3: {V['Q3']['verdict']}.**", ""]
    q4a = beh["ratings_vs_worth_r"]
    V["Q4a"] = {"r": q4a, "pass": q4a >= pr["q4a_corr"]}
    L_ += ["## Q4. Stated importance and the trade-offs", "",
           f"- **Q4a** stated ratings vs trade-off worth: r = {q4a:.2f} (rank {beh['ratings_vs_worth_spearman']:.2f}): **{'pass' if V['Q4a']['pass'] else 'fail'}**"]
    if (R / "rating.json").exists():
        rt = json.load(open(R / "rating.json"))
        V["Q4b"] = {"r": rt["q4b_corr_heldout"], "pass": rt["q4b_corr_heldout"] >= pr["q4b_corr"]}
        sh = rt["q4c_share"]
        testable = rt["site_swap_of_donor"]["value"] >= pr["q4c_site_min"]
        V["Q4c"] = {"share": sh["value"], "ci95": sh["ci95"], "testable": testable,
                    "pass": bool(testable and sh["value"] >= pr["q4c_share"] and sh["ci95"][0] > 0)}
        L_ += [f"- **Q4b** in the rating prompt (no comparison), the trade-off worth direction ({rt['site']} at layer {rt['layer']}) orders the "
               f"12 held-out values with r = {rt['q4b_corr_heldout']:.2f} against their trade-off worth (all 24: {rt['corr_all_worth']:.2f}; "
               f"against the stated ratings: {rt['corr_all_ratings']:.2f}): **{'pass' if V['Q4b']['pass'] else 'fail'}**",
               f"- **Q4c** swapping only that direction between rating prompts carries {sh['value']:.2f} [{sh['ci95'][0]:.2f}, {sh['ci95'][1]:.2f}] "
               f"of the whole site's effect on the stated rating ({rt['n_pairs']} pairs; the whole site carries "
               f"{rt['site_swap_of_donor']['value']:.2f} of the difference between the two ratings): "
               f"**{('pass' if V['Q4c']['pass'] else 'fail') if V['Q4c']['testable'] else 'not testable (the site carries too little of the rating)'}**"]
    L_ += [""]
    if (R / "source.json").exists():
        s = json.load(open(R / "source.json"))
        V["Q2a"] = {"r": s["q2a_corr_heldout"], "pass": s["q2a_corr_heldout"] >= pr["q2a_corr"]}
        L_ += ["## Q2. Does the same thing light up across trade-offs?", "",
               f"- **Q2a** the worth direction on option A's own words ({s['site']} at layer {s['layer']}; the whole site carries "
               f"{s['site_swap_share']:.2f} of a value swap), trained on 12 values, orders the 12 held-out values with r = "
               f"{s['q2a_corr_heldout']:.2f} (training values {s['corr_train']:.2f}); it carries "
               f"{s['heldout_swap']['of_site_swap']['value']:.2f} of the site's effect on held-out swaps: **{'pass' if V['Q2a']['pass'] else 'fail'}**"]
    if (R / "compare.json").exists():
        cmp_ = json.load(open(R / "compare.json"))
        V["Q2b"] = {"share": cmp_["q2b_share_heldout"], "pass": cmp_["q2b_share_heldout"] >= pr["q2b_share"]}
        L_ += [f"- **Q2b** the comparison direction ({cmp_['site']} at layer {cmp_['layer']}; the answer position arrives at layer "
               f"{cmp_['arrival_layer']}), trained on trade-offs among the 12 training values, carries "
               f"{cmp_['q2b_share_heldout']:.2f} of the site's effect on trade-offs among the 12 held-out values: "
               f"**{'pass' if V['Q2b']['pass'] else 'fail'}**"]
    (Path(cfg["res"]) / "report.md").write_text("\n".join(L_) + "\n")
    save_json(V, Path(cfg["res"]) / "verdicts8.json")
    log("report done: " + json.dumps({k: v.get("verdict", v.get("pass")) for k, v in V.items()}))


def main():
    args = parse_args()
    cfg = load_cfg(args)
    D = Data(cfg)
    stages = args.stages.split(",")
    name = list(cfg["models"])[0]
    log(f"Pilot 8; {len(D.rows)} trade-offs, {len(D.menus)} menu prompts, {len(D.ratings)} ratings, "
        f"{len(D.rating_pairs)} rating pairs; pairs: " + ", ".join(f"{k} {len(v)}" for k, v in D.sets.items()))
    if any(s != "report" for s in stages):
        M = p7.Model(cfg, name, D)
        for s in stages:
            if s == "report":
                continue
            t0 = time.time()
            log(f"===== stage {s} =====")
            {"behave": stage_behave, "source": stage_source, "compare": stage_compare, "rating": stage_rating}[s](M, D, cfg)
            log(f"===== stage {s} finished in {(time.time() - t0) / 60:.1f} min =====")
    if "report" in stages:
        stage_report(cfg, name, D)


if __name__ == "__main__":
    main()
