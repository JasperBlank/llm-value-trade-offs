"""Pilot 16: is the finished price absolute, or relative to the alternative? (DESIGN.md)

  python pilot16.py            Qwen3-14B: results/real/
  python pilot16.py --quick    Qwen3-0.6B on 6 values (24 sentences): results/quick/ (a pipeline test)

Every A (pilot 10's sentences and pilot 14's word deletions of them) against each of 8 alternatives B; the choice logit
and the states at the line breaks after A and after B and at the answer position (layers 16-32 by 4). The word effects
in the choice and in a price readout, and how much of them is common to all Bs (Q1-Q3); the report.
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
          "pilot16-relative-or-absolute"):
    sys.path.insert(0, str(ROOT / p))
import pilot11 as p11  # noqa: E402
import pilot14 as p14  # noqa: E402
import pilot15 as p15  # noqa: E402
import probe11 as pb  # noqa: E402
from common import batches, forward, log  # noqa: E402

POS = ["A_nl", "B_nl", "END"]
MAIN_POS, MAIN_LAYER = "B_nl", 24
SCALES = (0.01, 0.1, 1.0, 10.0)
Q_MIN = 0.9


class Runner16(p15.Runner15):
    def read_sel(self, encs, layers, bs=16):
        """States at POS after each layer in `layers` (float16), and logit P(A)."""
        torch = self.torch
        n = len(encs)
        X = np.zeros((n, len(POS), len(layers), self.d), dtype=np.float16)
        lg = np.zeros(n)
        for ch in batches(n, bs):
            ids, att, roles = self._batch_roles(encs, ch)
            cap = {}
            hooks = [self.blocks[L - 1].register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(li, o[0] if isinstance(o, tuple) else o))
                     for li, L in enumerate(layers)]
            try:
                with torch.no_grad():
                    lg[ch] = self._logit(forward(self.model, ids, att, torch))
            finally:
                for h in hooks:
                    h.remove()
            with torch.no_grad():
                for li in range(len(layers)):
                    for j, i in enumerate(ch):
                        X[i, :, li] = torch.stack([cap[li][j, roles[j][k]].float() for k in POS]).cpu().numpy().astype(np.float16)
        return X, lg


def common_share(T):
    """T: items x alternatives. Share of the item-related sum of squares that is the same for every alternative:
    SS(item means) / (SS(item means) + SS(interaction)), after removing each alternative's mean."""
    T = np.asarray(T, dtype=np.float64)
    g = T.mean()
    r = T.mean(1, keepdims=True) - g
    c = T.mean(0, keepdims=True) - g
    inter = T - g - r - c
    ss_item = T.shape[1] * float((r ** 2).sum())
    ss_int = float((inter ** 2).sum())
    return ss_item / (ss_item + ss_int)


def ridge_fit_predict(torch, dev, Xtr, ytr, groups, Xte):
    """Ridge on standardised states (linear kernel, on the GPU); the penalty chosen by leaving one value out on the
    training rows. Returns predictions for the training rows (in sample) and for Xte."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    K = A @ A.T
    y = torch.tensor(ytr, dtype=torch.float32, device=dev)
    d = Xtr.shape[1]
    errs = []
    for s in SCALES:
        err = 0.0
        for gv in np.unique(groups):
            te = torch.tensor(groups == gv, device=dev)
            tr = ~te
            ym = y[tr].mean()
            cf = torch.linalg.solve(K[tr][:, tr] + s * d * torch.eye(int(tr.sum()), device=dev), (y[tr] - ym)[:, None])[:, 0]
            err += float(((ym + K[te][:, tr] @ cf - y[te]) ** 2).sum())
        errs.append(err)
    s = SCALES[int(np.argmin(errs))]
    ym = y.mean()
    cf = torch.linalg.solve(K + s * d * torch.eye(len(ytr), device=dev), (y - ym)[:, None])[:, 0]
    out = []
    for ch in batches(len(Xte), 2048):
        B = torch.tensor((Xte[ch] - mu) / sd, dtype=torch.float32, device=dev)
        out.append((ym + (B @ A.T) @ cf).cpu().numpy())
    return (ym + K @ cf).cpu().numpy(), np.concatenate(out), s


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
    new = [d for d in json.load(open(ROOT / "pilot13-recipe-in-words/results/real/features13.json")) if d["set"] == "new"]
    Bs = p14.pick_partners(new, 8)
    uB = np.array([b["u"] for b in Bs])
    nb = len(Bs)
    M = Runner16(model_name)
    layers = [L for L in range(16, 33, 4) if L <= M.n_layers]
    li_main = layers.index(MAIN_LAYER) if MAIN_LAYER in layers else len(layers) // 2
    var_key, vtexts = [], []
    for i, s in enumerate(S):
        w = s["text"].split(" ")
        for k in range(len(w)):
            var_key.append((i, k))
            vtexts.append(" ".join(w[:k] + w[k + 1:]))
    parent = np.array([i for i, _ in var_key])
    nd = len(var_key)
    texts = [s["text"] for s in S] + vtexts
    log(f"{model_name}: {n} sentences, {nd} deletions, {nb} alternatives B (prices " + ", ".join(f"{x:+.1f}" for x in uB) + ")")

    # ---- every A against every B
    encs = [p15.encode_roles(M.tok, p11.menu_user(t, b["text"]), t, b["text"]) for b in Bs for t in texts]
    X, lg = M.read_sel(encs, layers)
    nt = len(texts)
    LG = lg.reshape(nb, nt).T                                  # texts x Bs
    L_int, L_del = LG[:n], LG[n:]
    dl = L_del - L_int[parent]                                 # deletions x Bs: the choice's word effects
    q2 = common_share(dl)
    q3 = common_share(L_int)
    log(f"choice: word effects common share {q2:.3f}; whole outcomes {q3:.3f}; median |word effect| {np.median(np.abs(dl)):.2f} logits")

    # ---- the readout's word effects at each position and layer (one ridge, trained on the intact menus of all Bs)
    rows_int = np.array([k * nt + i for k in range(nb) for i in range(n)])            # intact menus, B-major
    rows_del = np.array([k * nt + n + j for k in range(nb) for j in range(nd)])
    g_int = np.tile(groups, nb)
    shares, shares_mid, shares_perB, transfer = {}, {}, {}, {}
    mid = [k for k in range(nb) if k not in (int(np.argmin(uB)), int(np.argmax(uB)))]
    for pi, pname in enumerate(POS):
        for li, L in enumerate(layers):
            Xi = X[rows_int, pi, li].astype(np.float32)
            Xd = X[rows_del, pi, li].astype(np.float32)
            pin, pdel, s = ridge_fit_predict(M.torch, M.dev, Xi, lg[rows_int], g_int, Xd)
            P_int = pin.reshape(nb, n).T
            P_del = pdel.reshape(nb, nd).T
            dp = P_del - P_int[parent]
            key = f"{pname}@{L}"
            shares[key] = common_share(dp)
            shares_mid[key] = common_share(dp[:, mid])
            if pname == MAIN_POS:
                per = np.zeros((nd, nb))
                for k in range(nb):
                    rk = np.arange(k * n, (k + 1) * n)
                    pk_in, pk_del, _ = ridge_fit_predict(M.torch, M.dev, Xi[rk], lg[rows_int][rk], groups, Xd[k * nd:(k + 1) * nd])
                    per[:, k] = pk_del - pk_in[parent]
                shares_perB[key] = common_share(per)
                if li == li_main:
                    T = np.zeros((nb, nb))
                    for i_ in range(nb):
                        ri = np.arange(i_ * n, (i_ + 1) * n)
                        for j_ in range(nb):
                            _, pj, _ = ridge_fit_predict(M.torch, M.dev, Xi[ri], lg[rows_int][ri], groups,
                                                         np.concatenate([Xi[j_ * n:(j_ + 1) * n], Xd[j_ * nd:(j_ + 1) * nd]]))
                            T[i_, j_] = pb.r(pj[n:] - pj[:n][parent], dl[:, j_])
                    transfer = {"matrix": T.tolist(), "same_B_mean": float(np.mean(np.diag(T))),
                                "other_B_mean": float(T[~np.eye(nb, dtype=bool)].mean())}
                    dp_main, P_int_main = dp, P_int
        log(f"readout {pname}: " + ", ".join(f"{L}: {shares[f'{pname}@{L}']:.3f}" for L in layers))
    q1 = shares[f"{MAIN_POS}@{layers[li_main]}"]

    # ---- the shape of any dependence on B
    mean_dl = dl.mean(1)
    slope_dl = [float(np.polyfit(mean_dl, dl[:, k], 1)[0]) for k in range(nb)]
    mean_dp = dp_main.mean(1)
    slope_dp = [float(np.polyfit(mean_dp, dp_main[:, k], 1)[0]) for k in range(nb)]
    r_dl = [pb.r(dl[:, k], mean_dl) for k in range(nb)]
    a_main = L_int.mean(1)
    slope_out = [float(np.polyfit(a_main, L_int[:, k], 1)[0]) for k in range(nb)]

    # ---- save and report
    ok = lambda b: "absolute" if b else "relative"
    V = {"model": model_name, "n": n, "n_deletions": nd, "B": [b["text"] for b in Bs], "B_prices": uB.tolist(), "layers": layers,
         "Q1": {"common_share": q1, "pass": bool(q1 >= Q_MIN), "where": f"{MAIN_POS}@{layers[li_main]}"},
         "Q2": {"common_share": q2, "pass": bool(q2 >= Q_MIN)},
         "Q3": {"common_share": q3, "pass": bool(q3 >= Q_MIN)},
         "verdict": ("absolute (finished late)" if (q2 >= Q_MIN and q3 >= Q_MIN) else
                     "relative, already in the finished price" if q1 < Q_MIN else "relative, entering after the finished price"),
         "common_share_by_position": shares, "common_share_without_extreme_Bs": shares_mid,
         "choice_words_without_extreme_Bs": common_share(dl[:, mid]), "choice_outcomes_without_extreme_Bs": common_share(L_int[:, mid]),
         "common_share_readout_per_B": shares_perB, "transfer": transfer,
         "slopes_word_effects_choice": slope_dl, "slopes_word_effects_readout": slope_dp, "r_word_effects_with_mean": r_dl,
         "slopes_outcomes": slope_out, "mean_logit_by_B": L_int.mean(0).tolist(), "minutes": (time.time() - t0) / 60}
    json.dump(V, open(res / "verdicts16.json", "w"), indent=1)
    np.savez_compressed(res / "effects16.npz", logits_intact=L_int, logits_deleted=L_del, dl=dl, dp_main=dp_main, P_int_main=P_int_main,
                        var_key=np.array(var_key), u=u, uB=uB)
    tab = lambda d: ["| position | " + " | ".join(str(L) for L in layers) + " |", "|---" * (len(layers) + 1) + "|"] + [
        f"| {p} | " + " | ".join(f"{d[f'{p}@{L}']:.3f}" for L in layers) + " |" for p in POS]
    rep = [f"# Pilot 16 results ({model_name})", "",
           f"{n} sentences and their {nd} single-word deletions as option A, each against {nb} alternatives B (prices "
           + ", ".join(f"{x:+.1f}" for x in uB) + f"): {nb * nt} menus at 8:8. The median word effect is {np.median(np.abs(dl)):.2f} logits.", "",
           "## Pre-registered questions (bar 0.9 for \"absolute\")", "",
           f"- **Q1** inside: is A's finished price the same whatever B is? Common share of the readout's word effects at the line break after B, "
           f"layer {layers[li_main]}: {q1:.3f}: **{ok(q1 >= Q_MIN)}**",
           f"- **Q2** choice: does a word count the same whatever B is? Common share of the choice's word effects: {q2:.3f}: **{ok(q2 >= Q_MIN)}**",
           f"- **Q3** choice: are whole outcomes priced the same whatever B is? Common share of the 96 × 8 logits: {q3:.3f}: **{ok(q3 >= Q_MIN)}**",
           f"- **Verdict:** {V['verdict']}", "",
           "## Common share of the readout's word effects, by position and layer", "",
           "One ridge readout per position and layer, trained on the intact menus of all Bs to predict the choice logit. "
           "The line break after A comes before B and must give 1.", ""] + tab(shares) + [
           "", f"Without the two extreme Bs ({uB.min():+.1f} and {uB.max():+.1f}): choice words {V['choice_words_without_extreme_Bs']:.3f}, "
           f"whole outcomes {V['choice_outcomes_without_extreme_Bs']:.3f}; readout:", ""] + tab(shares_mid) + [
           "", "With a separate readout for each B (line break after B): " + ", ".join(f"{k} {v:.3f}" for k, v in shares_perB.items()) + ".", "",
           "## The shape of the dependence on B", "",
           "| B (price) | mean logit of the 96 | slope: whole outcomes | slope: word effects, choice | r with the mean word effect | slope: word effects, readout |",
           "|---|---|---|---|---|---|"]
    for k in np.argsort(uB):
        rep.append(f"| {Bs[k]['text']} ({uB[k]:+.1f}) | {L_int[:, k].mean():+.2f} | {slope_out[k]:.2f} | {slope_dl[k]:.2f} | {r_dl[k]:.2f} | {slope_dp[k]:.2f} |")
    if transfer:
        rep += ["", f"Transfer at the line break after B, layer {layers[li_main]}: a readout trained with one B predicts the choice's word effects "
                f"with that B at r {transfer['same_B_mean']:.2f} (mean) and with the other Bs at r {transfer['other_B_mean']:.2f}."]
    rep += ["", f"Run time {V['minutes']:.1f} min."]
    (res / "report.md").write_text("\n".join(rep) + "\n")
    log(f"Q1 {q1:.3f}, Q2 {q2:.3f}, Q3 {q3:.3f}: {V['verdict']}; report written; {V['minutes']:.1f} min")


if __name__ == "__main__":
    main()
