"""Pilot 8: the ingredients of worth (DESIGN.md).

  python pilot8.py                 Qwen3-14B, all stages
  python pilot8.py --quick         Qwen3-0.6B and a subset (a pipeline test)

Stages:
  behave  P(A) for every behaviour prompt; the outcomes' worth u fitted by maximum likelihood; the
          ingredients' effects (bootstrap intervals), held-out R² (W1)
  map     minimal pairs (one ingredient changed in option A), swapped at one site and layer: where each
          ingredient's contribution sits (descriptive)
  das     the worth direction at the A site chosen by the rule; each outcome's position along it; W2
  report  verdicts and report.md
Reuses pilot 7's sites, patching and distributed alignment search.
"""
from __future__ import annotations

import argparse
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
from common import Hooks, batches, deep_update, forward, layer_output, letter_ids, log, save_json  # noqa: E402

FEATS = ["time", "health", "frame_loss", "size_large", "children", "poverty"]
LABEL = {"time": "free time (vs money)", "health": "health (vs money)", "frame_loss": "prevented loss (vs gain)",
         "size_large": "large (vs small)", "children": "children (vs adults)", "poverty": "living in poverty"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stages", default="behave,map,das,report")
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


class Data:
    def __init__(self, cfg):
        lim = cfg.get("limit") or {}
        rng = np.random.default_rng(0)
        pairs = [json.loads(l) for l in open(HERE / "pairs.jsonl", encoding="utf-8")]
        self.sets = {}
        for s in sorted({p["set"] for p in pairs}):
            ps = [p for p in pairs if p["set"] == s]
            n = lim.get(s, lim.get("pairs"))
            if n and len(ps) > n:
                ps = [ps[i] for i in sorted(rng.choice(len(ps), size=n, replace=False))]
            self.sets[s] = ps
        rows = [json.loads(l) for l in open(HERE / "behaviour.jsonl", encoding="utf-8")]
        if lim.get("behaviour") and len(rows) > lim["behaviour"]:
            rows = [rows[i] for i in sorted(rng.choice(len(rows), size=lim["behaviour"], replace=False))]
        self.rows = rows
        O = json.load(open(HERE / "outcomes.json"))
        self.outcomes, self.train, self.test = O["outcomes"], O["train"], O["test"]
        self.ids = [o["id"] for o in self.outcomes]


def features(outcomes):
    return np.array([[o["domain"] == "time", o["domain"] == "health", o["frame"] == "loss", o["size"] == "large",
                      o["who"] == "children", o["need"] == "poverty"] for o in outcomes], dtype=float)


# --------------------------------------------------------------------------
# Fitting the worth
# --------------------------------------------------------------------------
def fit_worth(ia, ib, dlog, p, n_out, init=None):
    """P(A) = sigmoid(u_A - u_B + b dlog + c), u_0 = 0; maximum likelihood on soft labels p."""
    from scipy.optimize import minimize
    p = np.clip(p, 1e-6, 1 - 1e-6)

    def f(x):
        u = np.concatenate([[0.0], x[:n_out - 1]])
        b, c = x[n_out - 1], x[n_out]
        z = u[ia] - u[ib] + b * dlog + c
        q = 1 / (1 + np.exp(-z))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        loss = -np.sum(p * np.log(q) + (1 - p) * np.log(1 - q))
        g = q - p
        gu = np.bincount(ia, g, n_out) - np.bincount(ib, g, n_out)
        return loss, np.concatenate([gu[1:], [np.sum(g * dlog)], [np.sum(g)]])
    x0 = np.zeros(n_out + 1) if init is None else init
    r = minimize(f, x0, jac=True, method="L-BFGS-B")
    u = np.concatenate([[0.0], r.x[:n_out - 1]])
    return u, float(r.x[n_out - 1]), float(r.x[n_out]), r.x


def ingredient_regression(X, u):
    A = np.column_stack([np.ones(len(u)), X])
    beta = np.linalg.lstsq(A, u, rcond=None)[0]
    pred = A @ beta
    r2 = 1 - ((u - pred) ** 2).sum() / ((u - u.mean()) ** 2).sum()
    loo = np.zeros(len(u))
    for i in range(len(u)):
        m = np.arange(len(u)) != i
        bi = np.linalg.lstsq(A[m], u[m], rcond=None)[0]
        loo[i] = A[i] @ bi
    r2_loo = 1 - ((u - loo) ** 2).sum() / ((u - u.mean()) ** 2).sum()
    return beta, float(r2), float(r2_loo)


def biggest(beta6):
    d = dict(zip(FEATS, beta6))
    sizes = {"domain": max(abs(d["time"]), abs(d["health"]), abs(d["time"] - d["health"])), "frame": abs(d["frame_loss"]),
             "size": abs(d["size_large"]), "who": abs(d["children"]), "need": abs(d["poverty"])}
    return max(sizes, key=sizes.get), sizes


def stage_behave(M, D, cfg):
    import torch
    rows = D.rows
    lids = letter_ids(M.tok)
    pa = np.zeros(len(rows))
    with torch.no_grad():
        for ch in batches(len(rows), int(cfg["batch_size"])):
            ids, att, _ = M.batch([rows[i]["user"] for i in ch])
            o = forward(M.model, ids, att, torch)
            pr = torch.softmax(o.logits[:, -1].float(), -1)
            a, b = pr[:, lids[0]].sum(-1), pr[:, lids[1]].sum(-1)
            pa[ch] = (a / (a + b)).cpu().numpy()
    np.save(M.res / "behaviour_pA.npy", pa)
    idx = {k: i for i, k in enumerate(D.ids)}
    ia = np.array([idx[r["a"]] for r in rows])
    ib = np.array([idx[r["b"]] for r in rows])
    dlog = np.array([math.log(r["na"] / r["nb"]) for r in rows])
    n = len(D.ids)
    u, b, c, x = fit_worth(ia, ib, dlog, pa, n)
    z = u[ia] - u[ib] + b * dlog + c
    lo = np.log(np.clip(pa, 1e-4, 1 - 1e-4) / (1 - np.clip(pa, 1e-4, 1 - 1e-4)))
    X = features(D.outcomes)
    beta, r2, r2_loo = ingredient_regression(X, u)
    # interactions (reported): size and frame by domain
    Xi = np.column_stack([X, X[:, 3] * X[:, 0], X[:, 3] * X[:, 1], X[:, 2] * X[:, 0], X[:, 2] * X[:, 1]])
    _, r2_int, r2_int_loo = ingredient_regression(Xi, u)
    rng = np.random.default_rng(int(cfg["fit"]["seed"]))
    boots = []
    for _ in range(int(cfg["fit"]["bootstrap"])):
        s = rng.integers(0, len(rows), len(rows))
        ub, bb, _, _ = fit_worth(ia[s], ib[s], dlog[s], pa[s], n, init=x)
        bt, _, _ = ingredient_regression(X, ub)
        boots.append(np.concatenate([bt[1:], [bb]]))
    boots = np.array(boots)
    big, sizes = biggest(beta[1:])
    eff = {}
    for k, f in enumerate(FEATS):
        lo_, hi_ = np.quantile(boots[:, k], [0.025, 0.975])
        pe = np.exp(boots[:, k] / boots[:, -1])
        eff[f] = {"beta": float(beta[k + 1]), "ci95": [float(lo_), float(hi_)], "people_factor": float(math.exp(beta[k + 1] / b)),
                  "people_factor_ci95": [float(np.quantile(pe, 0.025)), float(np.quantile(pe, 0.975))]}
    out = {"n_prompts": len(rows), "u": dict(zip(D.ids, map(float, u))), "b": b, "bias_towards_A": c,
           "b_ci95": [float(np.quantile(boots[:, -1], 0.025)), float(np.quantile(boots[:, -1], 0.975))],
           "fit_corr_logodds": float(np.corrcoef(z, lo)[0, 1]),
           "fit_accuracy": float(((z > 0) == (pa > 0.5)).mean()),
           "intercept": float(beta[0]), "effects": eff, "r2_in_sample": r2, "r2_heldout": r2_loo,
           "r2_with_interactions": r2_int, "r2_with_interactions_heldout": r2_int_loo,
           "biggest": big, "ingredient_sizes": {k: float(v) for k, v in sizes.items()},
           "mean_P_letters": None}
    save_json(out, M.res / "behave.json")
    log(f"  behave: {len(rows)} prompts fitted")


# --------------------------------------------------------------------------
# Ingredient maps (pilot 7's map, over the minimal-pair sets)
# --------------------------------------------------------------------------
def stage_map(M, D, cfg):
    p7.MAP_SETS = {s[4:]: s for s in M.pairs if s.startswith("ing_")}
    p7.stage_map(M)
    log("  map done")


# --------------------------------------------------------------------------
# The worth direction (W2)
# --------------------------------------------------------------------------
def site_sweep(M, D, cfg):
    """The rule: among the A sites and layers up to the cap, where the whole-site swap carries the most."""
    ps = M.pairs["das_train"][: int(cfg["das"]["sweep_pairs"])]
    best, table = None, {}
    for site in cfg["map"]["a_sites"]:
        for L in [L for L in M.map_layers if L <= int(cfg["das"]["a_layer_max"])]:
            Dd = p7.prep_site(M, ps, site, L)
            t = p7.through_zero(Dd["y_s"] - Dd["y_r"], Dd["y_d"] - Dd["y_r"])["value"]
            table[f"{site}@{L}"] = t
            if best is None or t > best[2] + 1e-12:
                best = (site, L, t)
    return best, table


def coordinates(M, D, site, L, w):
    """Each outcome's position along w at the site (in option A, n = 4, both wordings averaged)."""
    import torch
    sys.path.insert(0, str(HERE))
    from make_prompts import user  # noqa: E402
    by_id = {o["id"]: o for o in D.outcomes}
    ctx = by_id[D.ids[0]]
    users, owners = [], []
    for o in D.outcomes:
        b = ctx if o["id"] != ctx["id"] else by_id[D.ids[1]]
        for wd in (0, 1):
            users.append(user(o, 4, b, 4, wd))
            owners.append(o["id"])
    vals = np.zeros(len(users))
    wt = torch.tensor(w, dtype=torch.float32, device=M.dev)
    with torch.no_grad():
        for ch in batches(len(users), int(M.cfg["batch_size"])):
            cap = {}
            with Hooks() as hk:
                hk.fwd(M.layers[L - 1], lambda m, i, o: cap.__setitem__("h", layer_output(o)))
                ids, att, es = M.batch([users[i] for i in ch])
                forward(M.model, ids, att, torch)
            T = ids.shape[1]
            for j, i in enumerate(ch):
                kk = torch.tensor(es[j]["sites"][site], device=M.dev)
                vals[i] = float((cap["h"][j, T + kk].float() @ wt).mean())
    return {oid: float(np.mean([v for v, ow in zip(vals, owners) if ow == oid])) for oid in D.ids}


def stage_das(M, D, cfg):
    beh = json.load(open(M.res / "behave.json"))
    u = beh["u"]
    (site, L, t), table = site_sweep(M, D, cfg)
    log(f"  das: site chosen by the rule")
    Dtr = p7.prep_site(M, M.pairs["das_train"], site, L)
    W, hist = p7.das_train(M, Dtr, site, L, int(cfg["das"]["seed"]))
    Dte = p7.prep_site(M, M.pairs["das_test"], site, L)
    ev = p7.das_eval(M, Dte, site, L, W)
    ev.pop("coord_change")
    w = W[0].cpu().numpy()
    coord = coordinates(M, D, site, L, w)
    tr = np.array([coord[i] for i in D.train]), np.array([u[i] for i in D.train])
    if np.corrcoef(*tr)[0, 1] < 0:
        coord = {k: -v for k, v in coord.items()}
        w = -w
    te_c, te_u = np.array([coord[i] for i in D.test]), np.array([u[i] for i in D.test])
    X = features(D.outcomes)
    c_all = np.array([coord[i] for i in D.ids])
    u_all = np.array([u[i] for i in D.ids])
    bc, r2c, _ = ingredient_regression(X, c_all)
    bu, _, _ = ingredient_regression(X, u_all)
    out = {"site": site, "layer": L, "site_swap_share": t, "sweep": table, "train_loss_by_epoch": hist,
           "heldout_swap": ev, "coordinates": coord,
           "w2a_corr_heldout": float(np.corrcoef(te_c, te_u)[0, 1]),
           "corr_all_48": float(np.corrcoef(c_all, u_all)[0, 1]),
           "coord_effects": dict(zip(FEATS, map(float, bc[1:]))), "worth_effects": dict(zip(FEATS, map(float, bu[1:]))),
           "w2b_weights_corr": float(np.corrcoef(bc[1:], bu[1:])[0, 1]), "coord_r2_ingredients": r2c}
    np.save(M.res / "worth_direction.npy", w.astype(np.float32))
    save_json(out, M.res / "das.json")
    log("  das done")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def stage_report(cfg, name):
    R = Path(cfg["res"]) / name
    pr = cfg["prereg"]
    beh = json.load(open(R / "behave.json"))
    V, lines = {}, [f"# Pilot 8 results ({cfg['models'][name]['path']})", ""]
    r2 = beh["r2_heldout"]
    w1 = ("worth is (mostly) a sum of ingredients" if r2 >= pr["w1_full"] else
          "partly" if r2 >= pr["w1_partial"] else "not explained by these ingredients")
    V["W1"] = {"r2_heldout": r2, "verdict": w1}
    lines += ["## Behaviour: what each ingredient adds", "",
              f"{beh['n_prompts']} prompts. Worth fitted per outcome (P(A) = σ(u_A − u_B + b·log(n_A/n_B) + c)): "
              f"b = {beh['b']:.2f} [{beh['b_ci95'][0]:.2f}, {beh['b_ci95'][1]:.2f}] per log person, bias towards A {beh['bias_towards_A']:+.2f}; "
              f"the fit orders the prompts' answers with r = {beh['fit_corr_logodds']:.2f} (log-odds), {beh['fit_accuracy']:.0%} same side.", "",
              "| ingredient | effect on worth (95%) | in people: counts as … × | (95%) |", "|---|---|---|---|"]
    for f in FEATS:
        e = beh["effects"][f]
        lines.append(f"| {LABEL[f]} | {e['beta']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}] | {e['people_factor']:.2f} | "
                     f"[{e['people_factor_ci95'][0]:.2f}, {e['people_factor_ci95'][1]:.2f}] |")
    lines += ["", f"Biggest ingredient: **{beh['biggest']}** (sizes: " + ", ".join(f"{k} {v:.2f}" for k, v in beh["ingredient_sizes"].items()) + ").",
              f"The ingredients added up explain R² = {beh['r2_in_sample']:.2f} of the 48 outcomes' worth (held out, one at a time: "
              f"**{r2:.2f}**); with size × domain and frame × domain: {beh['r2_with_interactions']:.2f} (held out {beh['r2_with_interactions_heldout']:.2f}).",
              "", f"**W1: {w1}.**", ""]
    if (R / "das.json").exists():
        das = json.load(open(R / "das.json"))
        a, bb = das["w2a_corr_heldout"], das["w2b_weights_corr"]
        w2 = bool(a >= pr["w2_corr"] and bb >= pr["w2_weights"])
        V["W2"] = {"site": das["site"], "layer": das["layer"], "heldout_corr": a, "weights_corr": bb,
                   "verdict": "assembled on the outcome's own words" if w2 else "not assembled there"}
        lines += ["## Inside the model: is the worth assembled on the outcome's own words?", "",
                  f"Site chosen by the rule: **{das['site']} at layer {das['layer']}** (its whole swap carries {das['site_swap_share']:.2f} of an "
                  f"outcome swap). The worth direction carries {das['heldout_swap']['of_site_swap']['value']:.2f} of the site's effect on held-out outcome swaps.", "",
                  f"- Held-out outcomes: position vs worth r = **{a:.2f}** (all 48: {das['corr_all_48']:.2f})",
                  f"- Ingredient weights, position vs worth: r = **{bb:.2f}**; the ingredients explain R² = {das['coord_r2_ingredients']:.2f} of the positions", "",
                  "| ingredient | effect on worth | effect on the position |", "|---|---|---|"]
        for f in FEATS:
            lines.append(f"| {LABEL[f]} | {das['worth_effects'][f]:+.2f} | {das['coord_effects'][f]:+.3f} |")
        lines += ["", f"**W2: {V['W2']['verdict']}.**", ""]
    if (R / "map.json").exists():
        mp = json.load(open(R / "map.json"))
        Ls = mp["layers"]
        lines += ["## Where each ingredient's contribution sits (descriptive)", ""]
        for ing, rec in mp["swaps"].items():
            lines += [f"**{ing}** ({rec['n_pairs']} minimal pairs; mean |effect| {rec['mean_abs_effect']:.2f}):", "",
                      "| site | " + " | ".join(str(L) for L in Ls) + " |", "|---|" + "---|" * len(Ls)]
            for s in mp["sites"]:
                lines.append(f"| {s} | " + " | ".join(f"{rec['transfer'][s][str(L)]['value']:.2f}" for L in Ls) + " |")
            lines.append("")
    (Path(cfg["res"]) / "report.md").write_text("\n".join(lines) + "\n")
    save_json(V, Path(cfg["res"]) / "verdicts8.json")
    log("report done: " + json.dumps({k: v["verdict"] for k, v in V.items()}))


def main():
    args = parse_args()
    cfg = load_cfg(args)
    D = Data(cfg)
    stages = args.stages.split(",")
    name = list(cfg["models"])[0]
    log(f"Pilot 8; {len(D.rows)} behaviour prompts; pairs: " + ", ".join(f"{k} {len(v)}" for k, v in D.sets.items()))
    if any(s != "report" for s in stages):
        M = p7.Model(cfg, name, D)
        for s in stages:
            if s == "report":
                continue
            t0 = time.time()
            log(f"===== stage {s} =====")
            {"behave": stage_behave, "map": stage_map, "das": stage_das}[s](M, D, cfg)
            log(f"===== stage {s} finished in {(time.time() - t0) / 60:.1f} min =====")
    if "report" in stages:
        stage_report(cfg, name)


if __name__ == "__main__":
    main()
