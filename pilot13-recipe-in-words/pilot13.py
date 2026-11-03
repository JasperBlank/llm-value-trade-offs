"""Pilot 13: the choice recipe in words (DESIGN.md).

  python pilot13.py            Qwen3-14B: results/real/
  python pilot13.py --quick    Qwen3-0.6B on 6 values (24 old and 6 new sentences): results/quick/ (a pipeline test)

Steps: the ten features of every sentence (the model's own 0-9 answers); the new sentences' ratings and prices (a new
menu against the old sentences); the states; the recipes (P1, P2, P4); the pushes at layer 20 (P3 and the feature
pushes); the report. Reuses pilot 11's model runner (reading states, pushing along a direction) and pilot 12's pushes.
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
          "pilot11-vibe-direction", "pilot12-recipes", "pilot13-recipe-in-words"):
    sys.path.insert(0, str(ROOT / p))
import build13 as b13  # noqa: E402
import pilot11 as p11  # noqa: E402
import probe11 as pb  # noqa: E402
import probe12 as q12  # noqa: E402
from build_prompts import WORDINGS  # noqa: E402  (pilot 9)
from build_prompts10 import BENEFICIARY, Q2, R2  # noqa: E402
from common import log  # noqa: E402

P1_MIN, P4_MIN, N_RANDOM, N_BOOT = 0.8, 0.7, 20, 2000
PUSH_LAYER, PUSH = 20, 2.0              # as pilot 12: layer 20, 2 price units along the price direction
SCALES = (0.01, 0.1, 1.0, 10.0)         # ridge penalties (times the state size), as pilots 11-12
N = b13.N_PEOPLE


def menu_user_n(a, na, b, nb):
    """Pilot 10's menu prompt with any numbers of people."""
    return "\n".join([BENEFICIARY, Q2, f"A) {na} people {a}", f"B) {nb} people {b}", R2])


def rating_user_w(text, w):
    """Pilot 10's rating prompt: pilot 9's wording w, 8 people."""
    return "\n".join([BENEFICIARY, WORDINGS[w][0], f"{N} people {text}", WORDINGS[w][1]])


def zcols(F, ref):
    return (F - ref.mean(0)) / (ref.std(0) + 1e-9)


def ols(Ftr, ytr, Fte):
    """Least squares with an intercept: predictions for Fte, and the coefficients (intercept first)."""
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(Ftr)), Ftr]), ytr, rcond=None)[0]
    return np.column_stack([np.ones(len(Fte)), Fte]) @ beta, beta


def ols_lovo(F, y, groups):
    """Held-out predictions, leaving out one value (all its sentences) at a time."""
    pred = np.zeros(len(y))
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        pred[te] = ols(F[tr], y[tr], F[te])[0]
    return pred


def std_weights(F, y):
    """Standardised weights: z(y) on the z-scored features, fitted together."""
    Z = zcols(F, F)
    return ols(Z, (y - y.mean()) / y.std(), Z)[1][1:]


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(np.asarray(v, dtype=float)))
    return pb.r(rank(x), rank(y))


def fit_new_prices(logit, rows, u_old, b, c):
    """Each new sentence's price by maximum likelihood on the soft answers, with the old prices, b and the A-bias held
    fixed: P(A) = sigmoid(u_A - u_B + b log(nA/nB) + c). rows: (new j, old k, 1 if the new one is A, nA, nB)."""
    from scipy.optimize import minimize_scalar
    p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    sign = np.where(rows[:, 2] == 1, 1.0, -1.0)
    rest = -sign * u_old[rows[:, 1]] + b * np.log(rows[:, 3] / rows[:, 4]) + c
    out = np.zeros(int(rows[:, 0].max()) + 1)
    for j in range(len(out)):
        m = rows[:, 0] == j
        nll = lambda x, m=m: float(np.sum(p[m] * np.logaddexp(0, -(sign[m] * x + rest[m]))
                                          + (1 - p[m]) * np.logaddexp(0, sign[m] * x + rest[m])))
        out[j] = minimize_scalar(nll, bounds=(-25, 25), method="bounded").x
    return out


def ridge_new(Xo, y, Xn, groups):
    """A ridge readout of the states (standardised on the old sentences; linear kernel), its penalty chosen by
    leave-one-value-out on the old sentences. Returns its held-out r on the old ones, its predictions for the new ones,
    and the chosen penalty (in units of the state size)."""
    mu, sd = Xo.mean(0), Xo.std(0) + 1e-6
    Ao, An = (Xo - mu) / sd, (Xn - mu) / sd
    K, d = Ao @ Ao.T, Xo.shape[1]
    errs = []
    for s in SCALES:
        err = 0.0
        for g in np.unique(groups):
            te, tr = groups == g, groups != g
            err += np.sum((q12._ridge_k(K, y, tr, te, s * d) - y[te]) ** 2)
        errs.append(err)
    s = SCALES[int(np.argmin(errs))]
    heldout = q12.lovo_ridge_K(K, y, groups, d, SCALES)
    coef = np.linalg.solve(K + s * d * np.eye(len(y)), y - y.mean())
    return pb.r(heldout, y), y.mean() + An @ Ao.T @ coef, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    name = "Qwen/Qwen3-0.6B" if a.quick else "Qwen/Qwen3-14B"
    res = HERE / "results" / ("quick" if a.quick else "real")
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- material
    S = p11.items(a.quick)              # pilot 10's sentences, with price, W0 rating and the 4 nearest-priced partners of other values
    vals = sorted({s["value"] for s in S})
    NEW = [n for n in json.load(open(HERE / "sentences13.json"))["new"] if n["value"] in vals]
    P10 = ROOT / "pilot10-full-range"
    idx10 = {s["text"]: i for i, s in enumerate(json.load(open(P10 / "sentences10.json"))["sentences"])}
    R10 = json.load(open(P10 / "results/real/ratings10.json"))
    rate10 = {tuple(k): x for k, x in zip(R10["keys"], R10["rating"])}
    menu10 = json.load(open(P10 / "results/real/verdicts10.json"))["menu"]
    b10, c10 = menu10["b"], menu10["bias_towards_A"]
    u = np.array([s["u"] for s in S])
    Rbar = np.array([np.mean([rate10[(idx10[s["text"]], N, w)] for w in (0, 1, 2)]) for s in S])
    groups = np.array([vals.index(s["value"]) for s in S])
    feat = [f[0] for f in b13.FEATURES]
    texts = [s["text"] for s in S] + [n["text"] for n in NEW]
    n_old, n_new = len(S), len(NEW)
    M = p11.Runner(name)
    er = lambda LP: np.exp(LP) @ np.arange(10)
    log(f"{name}: {n_old} old and {n_new} new sentences")

    # ---- 1. the ten features: the model's own answers, as expected digits
    encs = [p11.encode(M.tok, b13.feature_user(t, f), f"\n{N} people ", t) for f in b13.FEATURES for t in texts]
    F = er(M.rating(encs)).reshape(len(feat), len(texts)).T          # sentences x features
    F_old, F_new = F[:n_old], F[n_old:]
    log("features done; r with the price: " + ", ".join(f"{n} {pb.r(F_old[:, i], u):+.2f}" for i, n in enumerate(feat)))

    # ---- 2. the new sentences: 'how good' ratings (W0-W2, 8 people), and prices from a menu against the old sentences
    renc = [p11.encode(M.tok, rating_user_w(n["text"], w), f"\n{N} people ", n["text"]) for w in (0, 1, 2) for n in NEW]
    Rbar_new = er(M.rating(renc)).reshape(3, n_new).mean(0)
    rows, menc = [], []
    for j, n in enumerate(NEW):
        for k, s in enumerate(S):
            for na, nb in b13.NEW_PAIRS_N:
                rows.append((j, k, 1, na, nb))
                menc.append(p11.encode(M.tok, menu_user_n(n["text"], na, s["text"], nb), f"A) {na} people ", n["text"]))
                rows.append((j, k, 0, na, nb))
                menc.append(p11.encode(M.tok, menu_user_n(s["text"], na, n["text"], nb), f"A) {na} people ", s["text"]))
    rows = np.array(rows)
    new_logit = M.menu_logit(menc)
    u_new = fit_new_prices(new_logit, rows, u, b10, c10)
    np.savez_compressed(res / "new_menu13.npz", logit=new_logit.astype(np.float32), rows=rows)
    log(f"new prices done ({len(rows)} trade-offs); range {u_new.min():+.2f} to {u_new.max():+.2f} (old {u.min():+.2f} to {u.max():+.2f})")

    # ---- 3. states: the mean over the sentence's tokens, as option A (before B is read) and in the W0 rating prompt
    layers = list(range(0, M.n_layers + 1, 4))
    menc_s = ([p11.encode(M.tok, p11.menu_user(s["text"], S[s["partners"][0]]["text"]), f"A) {N} people ", s["text"]) for s in S]
              + [p11.encode(M.tok, p11.menu_user(n["text"], S[0]["text"]), f"A) {N} people ", n["text"]) for n in NEW])
    renc_s = [p11.encode(M.tok, p11.rating_user(t), f"\n{N} people ", t) for t in texts]
    Xm, _ = M.read(menc_s, layers)
    Xr, _ = M.read(renc_s, layers)
    np.savez_compressed(res / "states13.npz", menu_mean=Xm.astype(np.float16), rating_mean=Xr.astype(np.float16), layers=np.array(layers),
                        n_old=n_old, features=F, u_old=u, u_new=u_new, Rbar_old=Rbar, Rbar_new=Rbar_new)
    log("states done")

    # ---- 4. the recipes (P1, P2, P4)
    W = slice(1, None)                                   # F2-F10: the recipe in words
    p1 = pb.r(ols_lovo(F_old[:, W], u, groups), u)
    p1_all = pb.r(ols_lovo(F_old, u, groups), u)
    p1_f1 = pb.r(ols_lovo(F_old[:, :1], u, groups), u)
    r_rating_words = pb.r(ols_lovo(F_old[:, W], Rbar, groups), Rbar)
    w_u, w_R = std_weights(F_old[:, W], u), std_weights(F_old[:, W], Rbar)
    w_u_all, w_R_all = std_weights(F_old, u), std_weights(F_old, Rbar)
    rng = np.random.default_rng(13)
    boots = np.zeros((N_BOOT, len(b13.WORDS)))
    for k in range(N_BOOT):
        i = rng.integers(0, n_old, n_old)
        boots[k] = std_weights(F_old[i][:, W], Rbar[i]) - std_weights(F_old[i][:, W], u[i])
    diff_ci = np.quantile(boots, [0.025, 0.975], axis=0)
    mi = b13.WORDS.index("moral")
    p2_est, p2_ci = float(w_R[mi] - w_u[mi]), [float(diff_ci[0, mi]), float(diff_ci[1, mi])]
    p2 = bool(p2_ci[0] > 0)
    pred_new = ols(F_old[:, W], u, F_new[:, W])[0]
    p4 = pb.r(pred_new, u_new)
    li = layers.index(PUSH_LAYER) if PUSH_LAYER in layers else len(layers) // 2
    L = layers[li]
    ridge_r_old, pred_new_states, ridge_s = ridge_new(Xm[:n_old, li].astype(np.float64), u, Xm[n_old:, li].astype(np.float64), groups)
    new_preds = {"recipe_in_words": p4, "with_F1": pb.r(ols(F_old, u, F_new)[0], u_new), "F1_alone": pb.r(F_new[:, 0], u_new),
                 f"states_ridge_layer{L}": pb.r(pred_new_states, u_new), "how_good_rating": pb.r(Rbar_new, u_new)}
    log(f"recipes: P1 r {p1:.2f}; P2 {p2_est:+.2f} [{p2_ci[0]:+.2f}, {p2_ci[1]:+.2f}]; P4 r {p4:.2f}")

    # ---- 5. pushes at layer 20, set up as in pilot 12 (P3, and the feature pushes)
    Xp = Xm[:n_old, li].astype(np.float64)
    v_price, _, c = pb.full_dom(Xp, u)
    dl = PUSH / c
    dis = q12.cubic_residual(u, Rbar)
    dirs = {"price": v_price, "rating": pb.dom_direction(Xp, Rbar), "disagreement": pb.dom_direction(Xp, dis)}
    for i, n in enumerate(feat):
        dirs[f"feature:{n}"] = pb.dom_direction(Xp, F_old[:, i])
    rr = np.random.default_rng(1300)
    for k in range(N_RANDOM):
        v = rr.normal(size=M.d)
        dirs[f"random:{k}"] = v / np.linalg.norm(v)
    menu_items = [(i, j) for i, s in enumerate(S) for j in s["partners"]]
    menu_all = [p11.encode(M.tok, p11.menu_user(S[i]["text"], S[j]["text"]), f"A) {N} people ", S[i]["text"]) for i, j in menu_items]
    rate_all = [p11.encode(M.tok, p11.rating_user(s["text"]), f"\n{N} people ", s["text"]) for s in S]
    owner = np.array([i for i, _ in menu_items])
    pushes = {}
    for kname, vec in dirs.items():
        mp = M.menu_logit(menu_all, (L, vec, np.full(len(menu_all), dl)))
        mm = M.menu_logit(menu_all, (L, vec, np.full(len(menu_all), -dl)))
        rp = M.rating(rate_all, (L, vec, np.full(len(rate_all), dl)))
        rm = M.rating(rate_all, (L, vec, np.full(len(rate_all), -dl)))
        Cs = np.array([np.mean((mp - mm)[owner == i]) for i in range(n_old)]) / (2 * dl)     # logit per residual unit
        Rs = (er(rp) - er(rm)) / (2 * dl)                                                   # rating points per residual unit
        pushes[kname] = {"C": Cs, "R": Rs}
        log(f"push {kname}: choice {Cs.mean() * dl:+.3f} logit and rating {Rs.mean() * dl:+.3f} points per push")
    sr = pushes["price"]["R"].mean() / pushes["price"]["C"].mean()
    excess = {k: float(v["R"].mean() - sr * v["C"].mean()) for k, v in pushes.items()}
    rand = np.array([excess[f"random:{k}"] for k in range(N_RANDOM)])
    p3 = bool(excess["disagreement"] > rand.max())
    p3_rank_p = float((1 + np.sum(rand >= excess["disagreement"])) / (1 + N_RANDOM))
    _, dis_ci = q12.bootstrap_excess(pushes["price"]["C"], pushes["price"]["R"], pushes["disagreement"]["C"], pushes["disagreement"]["R"], N_BOOT, 13)
    randC = np.array([pushes[f"random:{k}"]["C"].mean() * dl for k in range(N_RANDOM)])
    randR = np.array([pushes[f"random:{k}"]["R"].mean() * dl for k in range(N_RANDOM)])
    r_feat_u = [pb.r(F_old[:, i], u) for i in range(len(feat))]
    r_feat_R = [pb.r(F_old[:, i], Rbar) for i in range(len(feat))]
    sp_choice = spearman([pushes[f"feature:{n}"]["C"].mean() for n in feat], r_feat_u)
    sp_excess = spearman([excess[f"feature:{n}"] for n in b13.WORDS], w_R - w_u)

    # ---- 6. save and report
    V = {"model": name, "n_old": n_old, "n_new": n_new, "n_new_tradeoffs": int(len(rows)),
         "P1": {"r": p1, "pass": bool(p1 >= P1_MIN), "r_with_F1": p1_all, "r_F1_alone": p1_f1, "rating_from_words_r": r_rating_words},
         "P2": {"moral_rating_minus_price": p2_est, "ci95": p2_ci, "pass": p2},
         "P3": {"disagreement_excess": excess["disagreement"], "ci95_over_sentences": dis_ci, "random_excess": rand.tolist(),
                "rank_p": p3_rank_p, "pass": p3},
         "P4": {"r": p4, "pass": bool(p4 >= P4_MIN), "other_predictors_r": new_preds, "ridge_heldout_r_on_old": ridge_r_old,
                "ridge_penalty_scale": ridge_s},
         "weights_words": {n: {"price": float(w_u[i]), "rating": float(w_R[i]), "rating_minus_price": float(w_R[i] - w_u[i]),
                               "diff_ci95": [float(diff_ci[0, i]), float(diff_ci[1, i])]} for i, n in enumerate(b13.WORDS)},
         "weights_with_F1": {n: {"price": float(w_u_all[i]), "rating": float(w_R_all[i])} for i, n in enumerate(feat)},
         "feature_r": {n: {"price": r_feat_u[i], "rating": r_feat_R[i]} for i, n in enumerate(feat)},
         "pushes": {k: {"choice_logit_per_push": float(v["C"].mean() * dl), "rating_points_per_push": float(v["R"].mean() * dl),
                        "excess_per_residual_unit": excess[k], "cos_with_price_direction": float(dirs[k] @ v_price)}
                    for k, v in pushes.items()},
         "push_layer": L, "delta_residual": float(dl), "rating_per_logit_price_direction": float(sr),
         "feature_pushes_choice_vs_price_r_spearman": sp_choice, "feature_excess_vs_weight_difference_spearman": sp_excess,
         "minutes": (time.time() - t0) / 60}
    json.dump(V, open(res / "verdicts13.json", "w"), indent=1)
    json.dump([dict(value=s["value"], level=s["level"], text=s["text"], set="old", u=float(u[i]), Rbar=float(Rbar[i]),
                    features=dict(zip(feat, map(float, F_old[i])))) for i, s in enumerate(S)]
              + [dict(value=n["value"], level=n["level"], text=n["text"], set="new", u=float(u_new[j]), Rbar=float(Rbar_new[j]),
                      u_from_words=float(pred_new[j]), u_from_states=float(pred_new_states[j]), features=dict(zip(feat, map(float, F_new[j]))))
                 for j, n in enumerate(NEW)], open(res / "features13.json", "w"), indent=1)

    ok = lambda b: "pass" if b else "fail"
    rep = [f"# Pilot 13 results ({name})", "",
           f"{n_old} old sentences (pilot 10's prices and ratings) and {n_new} new ones, priced here against the old ({len(rows)} trade-offs). "
           "Features: the model's own 0-9 answers to ten descriptive questions (expected digit).", "",
           "## Pre-registered questions", "",
           f"- **P1** the price from the nine features in words (F2-F10), held out by value: r = {p1:.2f}: **{ok(p1 >= P1_MIN)}** (bar {P1_MIN}). "
           f"With F1 (size) added: {p1_all:.2f}; F1 alone: {p1_f1:.2f}. The rating from the same nine: {r_rating_words:.2f}.",
           f"- **P2** 'putting right something morally wrong' weighs {p2_est:+.2f} more (standardised) in the rating than in the price, "
           f"95% interval [{p2_ci[0]:+.2f}, {p2_ci[1]:+.2f}]: **{ok(p2)}** (the interval must lie above 0)",
           f"- **P3** pilot 12's disagreement push: excess {excess['disagreement']:+.4f} rating points per residual unit "
           f"[{dis_ci[0]:+.4f}, {dis_ci[1]:+.4f}] over sentences; the 20 random directions {rand.min():+.4f} to {rand.max():+.4f} "
           f"(rank p = {p3_rank_p:.3f}): **{ok(p3)}** (must beat all 20)",
           f"- **P4** the {n_new} new sentences, the recipe's predicted prices against their measured prices: r = {p4:.2f}: **{ok(p4 >= P4_MIN)}** "
           f"(bar {P4_MIN})", "",
           "## The two recipes", "",
           "Standardised weights of F2-F10, fitted together on the old sentences; r is each feature's plain correlation.", "",
           "| feature | price (choice) | rating (stated) | rating minus price (95% interval) | r with price | r with rating |",
           "|---|---|---|---|---|---|"]
    for i, n in enumerate(b13.WORDS):
        w = V["weights_words"][n]
        rep.append(f"| {n} | {w['price']:+.2f} | {w['rating']:+.2f} | {w['rating_minus_price']:+.2f} [{w['diff_ci95'][0]:+.2f}, {w['diff_ci95'][1]:+.2f}] | "
                   f"{r_feat_u[i + 1]:+.2f} | {r_feat_R[i + 1]:+.2f} |")
    rep += [f"| size (F1; not in the recipe) | | | | {r_feat_u[0]:+.2f} | {r_feat_R[0]:+.2f} |", "",
            "With F1 added (F1-F10 together), price: " + ", ".join(f"{n} {w_u_all[i]:+.2f}" for i, n in enumerate(feat)) + ".",
            "Rating: " + ", ".join(f"{n} {w_R_all[i]:+.2f}" for i, n in enumerate(feat)) + ".", "",
            f"## Pushes at layer {L}", "",
            f"Every direction is a unit vector pushed by ±{dl:.2f} residual units (2 price units along the price direction), on the "
            f"sentence's tokens: as option A against its 4 partners for the choice, in the W0 prompt for the rating. The price direction "
            f"moves the rating {sr:.3f} points per logit of choice; excess = rating change minus that ratio times the choice change "
            f"(per residual unit; multiply by {dl:.2f} for per push).", "",
            "| direction | cos with price direction | choice: logit P(A) per push | rating: points per push | excess |", "|---|---|---|---|---|"]
    for k, v in V["pushes"].items():
        if not k.startswith("random:"):
            rep.append(f"| {k} | {v['cos_with_price_direction']:+.2f} | {v['choice_logit_per_push']:+.3f} | {v['rating_points_per_push']:+.3f} | "
                       f"{v['excess_per_residual_unit']:+.4f} |")
    rep += [f"| 20 random directions (range) | | {randC.min():+.3f} to {randC.max():+.3f} | {randR.min():+.3f} to {randR.max():+.3f} | "
            f"{rand.min():+.4f} to {rand.max():+.4f} |", "",
            f"- The random benchmark: the price direction moves the choice {V['pushes']['price']['choice_logit_per_push']:+.3f} logits per push; "
            f"the largest random direction {np.abs(randC).max():.3f}.",
            f"- Do the feature pushes move the choice in line with each feature's correlation with the price? Spearman {sp_choice:+.2f} (10 features).",
            f"- Does each feature's rating excess follow how much more the rating recipe weighs it? Spearman {sp_excess:+.2f} (9 features).", "",
            "## The new sentences", "",
            "Predictors of their measured prices (r): " + ", ".join(f"{k} {v:.2f}" for k, v in new_preds.items())
            + f". The states readout's held-out r on the old sentences: {ridge_r_old:.2f}.", "",
            "| value | sentence | measured price | from the words | from the states | how good (W0-W2) |", "|---|---|---|---|---|---|"]
    for j in np.argsort(-u_new):
        rep.append(f"| {NEW[j]['value']} | {NEW[j]['text']} | {u_new[j]:+.2f} | {pred_new[j]:+.2f} | {pred_new_states[j]:+.2f} | {Rbar_new[j]:.2f} |")
    rep += ["", f"Run time {V['minutes']:.1f} min."]
    (res / "report.md").write_text("\n".join(rep) + "\n")
    log(f"report written; {V['minutes']:.1f} min")


if __name__ == "__main__":
    main()
