"""Fitting the choices: one value per good, exchange rates, quantity slopes.

Two shapes for how value grows with the number of people (q):
  log:     P(first) = sigmoid( b*(log qX - log qY) + uX - uY + c )
           1 unit of X is worth exp((uX - uY)/b) units of Y.
  linear:  P(first) = sigmoid( exp(wX)*qX - exp(wY)*qY + c )
           1 unit of X is worth exp(wX - wY) units of Y.
c absorbs any preference for the first-listed letter. The targets are the model's exact
choice probabilities (soft labels), so the fit minimises cross-entropy against them.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from common import option_key

EPS = 1e-4


def two_option_table(rows: list[dict]) -> list[dict]:
    """One record per two-option prompt: first/second option, quantities, P(first)."""
    out = []
    for r in rows:
        if len(r["options"]) != 2:
            continue
        pa, pb = r["probs"][0], r["probs"][1]
        if pa + pb <= 0:
            continue
        o1, o2 = r["options"]
        out.append({"x": option_key(o1), "y": option_key(o2), "qx": o1["q"], "qy": o2["q"],
                    "p": pa / (pa + pb), "kind": r["kind"], "phrasing": r["phrasing"],
                    "beneficiary": r["beneficiary"], "goal_good": r.get("goal_good"),
                    "goods": (o1["good"], o2["good"]), "frame_": r["frame"], "mass": r["mass"]})
    return out


def _sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


class UtilityFit:
    def __init__(self, recs: list[dict], shape: str):
        self.shape = shape
        self.keys = sorted({r["x"] for r in recs} | {r["y"] for r in recs})
        self.idx = {k: i for i, k in enumerate(self.keys)}
        xi = np.array([self.idx[r["x"]] for r in recs])
        yi = np.array([self.idx[r["y"]] for r in recs])
        qx = np.array([r["qx"] for r in recs], dtype=float)
        qy = np.array([r["qy"] for r in recs], dtype=float)
        p = np.clip(np.array([r["p"] for r in recs]), EPS, 1 - EPS)
        K = len(self.keys)

        def unpack(th):
            u = np.concatenate([[0.0], th[:K - 1]])      # the first good is the reference (value 0)
            return u, th[K - 1], th[K]                    # u (or w), b (log only), c

        def loss(th):
            u, b, c = unpack(th)
            if shape == "log":
                z = b * (np.log(qx) - np.log(qy)) + u[xi] - u[yi] + c
            else:
                z = np.exp(u[xi]) * qx - np.exp(u[yi]) * qy + c
            s = _sig(z)
            L = -np.sum(p * np.log(s + 1e-12) + (1 - p) * np.log(1 - s + 1e-12))
            g = s - p
            gu = np.zeros(K)
            if shape == "log":
                np.add.at(gu, xi, g)
                np.add.at(gu, yi, -g)
                gb = np.sum(g * (np.log(qx) - np.log(qy)))
            else:
                np.add.at(gu, xi, g * np.exp(u[xi]) * qx)
                np.add.at(gu, yi, -g * np.exp(u[yi]) * qy)
                gb = 0.0
            return L, np.concatenate([gu[1:], [gb, np.sum(g)]])

        th0 = np.zeros(K + 1)
        th0[K - 1] = 1.0 if shape == "log" else 0.0
        if shape == "linear":
            th0[:K - 1] = -1.0
        res = minimize(loss, th0, jac=True, method="L-BFGS-B")
        self.u, self.b, self.c = unpack(res.x)
        self.n = len(recs)
        self.loss = float(res.fun)

    def z(self, recs):
        xi = np.array([self.idx[r["x"]] for r in recs])
        yi = np.array([self.idx[r["y"]] for r in recs])
        qx = np.array([r["qx"] for r in recs], dtype=float)
        qy = np.array([r["qy"] for r in recs], dtype=float)
        if self.shape == "log":
            return self.b * (np.log(qx) - np.log(qy)) + self.u[xi] - self.u[yi] + self.c
        return np.exp(self.u[xi]) * qx - np.exp(self.u[yi]) * qy + self.c

    def predict(self, recs):
        return _sig(self.z(recs))

    def rate(self, x: str, y: str) -> float:
        """How many units of y one unit of x is worth."""
        ux, uy = self.u[self.idx[x]], self.u[self.idx[y]]
        if self.shape == "log":
            return float(np.exp((ux - uy) / self.b)) if abs(self.b) > 1e-6 else float("nan")
        return float(np.exp(ux - uy))


def cv_correlation(recs: list[dict], shape: str) -> tuple[float, int]:
    """Leave one pair of goods out at a time, predict it from the rest; correlation of
    predicted and actual probabilities over all held-out prompts."""
    pairs = sorted({tuple(sorted((r["x"], r["y"]))) for r in recs})
    preds, obs = [], []
    for pr in pairs:
        held = [r for r in recs if tuple(sorted((r["x"], r["y"]))) == pr]
        train = [r for r in recs if tuple(sorted((r["x"], r["y"]))) != pr]
        train_keys = {r["x"] for r in train} | {r["y"] for r in train}
        if not train or pr[0] not in train_keys or pr[1] not in train_keys:
            continue
        fit = UtilityFit(train, shape)
        preds += list(fit.predict(held))
        obs += [r["p"] for r in held]
    if len(preds) < 3 or np.std(preds) == 0 or np.std(obs) == 0:
        return float("nan"), len(pairs)
    return float(np.corrcoef(preds, obs)[0, 1]), len(pairs)


def pair_rate(recs: list[dict], x: str, y: str, shape: str) -> float:
    """Exchange rate of x in units of y, fitted from the x–y prompts alone."""
    sub = [r for r in recs if {r["x"], r["y"]} == {x, y}]
    if not sub:
        return float("nan")
    return UtilityFit(sub, shape).rate(x, y)


def quantity_slope(recs: list[dict], target: str, shape: str):
    """How strongly the chance of choosing `target` rises with its own quantity.
    OLS of logit P(target chosen) on f(q_target), with fixed effects for the partner good and
    its quantity, the position of the target and the phrasing. Returns (slope, SE, n)."""
    rows = []
    for r in recs:
        if target not in (r["x"], r["y"]):
            continue
        first = r["x"] == target
        q_t, q_o, other = (r["qx"], r["qy"], r["y"]) if first else (r["qy"], r["qx"], r["x"])
        p_t = r["p"] if first else 1 - r["p"]
        rows.append((q_t, f"{other}:{q_o}", int(first), r["phrasing"], np.log(np.clip(p_t, 1e-3, 1 - 1e-3) /
                                                                             (1 - np.clip(p_t, 1e-3, 1 - 1e-3)))))
    if len(rows) < 6:
        return float("nan"), float("nan"), len(rows)
    q = np.array([r[0] for r in rows], dtype=float)
    f = np.log(q) if shape == "log" else q
    cols = [f, np.ones(len(rows))]
    for j in (1, 2, 3):
        levels = sorted({r[j] for r in rows}, key=str)
        for lev in levels[1:]:
            cols.append(np.array([1.0 if r[j] == lev else 0.0 for r in rows]))
    X = np.column_stack(cols)
    y = np.array([r[4] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    sigma2 = resid @ resid / dof
    try:
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(max(cov[0, 0], 0)))
    except np.linalg.LinAlgError:
        se = float("nan")
    return float(beta[0]), se, len(rows)


def transitivity_factor(recs: list[dict], trio: list[str], shape: str) -> float:
    """For goods a, b, c: how far the a-to-c exchange rate (fitted from a–c prompts alone)
    is from a-to-b times b-to-c (each fitted from its own prompts). 1.0 = perfectly transitive."""
    a, b, c = trio
    r_ab, r_bc, r_ac = pair_rate(recs, a, b, shape), pair_rate(recs, b, c, shape), pair_rate(recs, a, c, shape)
    if not all(np.isfinite([r_ab, r_bc, r_ac])) or min(r_ab, r_bc, r_ac) <= 0:
        return float("nan")
    return float(np.exp(abs(np.log(r_ac) - np.log(r_ab) - np.log(r_bc))))
