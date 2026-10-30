"""Cells, the utility model, exchange rates, quantity slopes, transitivity.

A CELL is one comparison: two options (good, wording, people) at given quantities, for one
beneficiary group. Its choice probability is the mean over both option orders and all
wordings, which removes any preference for a letter by design. Cells are stored with the
options in a fixed (sorted) orientation, so the utility model needs no letter term.

Value function (log, chosen in pilot 2):
    P(x chosen over y) = sigmoid( b * (log qx - log qy) + ux - uy )
One unit of x is worth exp((ux - uy) / b) units of y.
"""
from __future__ import annotations

import collections

import numpy as np
from scipy.optimize import minimize

from common import option_key

EPS = 1e-4


def cells(rows: list[dict], kinds=None, wordings=None, by_wording: bool = False) -> list[dict]:
    """Two-option prompts averaged into cells (over both orders, and over wordings unless by_wording)."""
    acc = collections.defaultdict(list)
    for r in rows:
        if len(r["options"]) != 2 or (kinds and r["kind"] not in kinds):
            continue
        if wordings is not None and r["phrasing"] not in wordings:
            continue
        pa, pb = r["probs"][0], r["probs"][1]
        if pa + pb <= 0:
            continue
        o1, o2 = r["options"]
        k1, k2 = (option_key(o1), o1["q"]), (option_key(o2), o2["q"])
        first, second = sorted([k1, k2])
        p_first = pa / (pa + pb) if k1 == first else pb / (pa + pb)
        key = (first, second, r["beneficiary"], r["kind"], r.get("goal_good"), r["phrasing"] if by_wording else None)
        acc[key].append((p_first, k1 == first))
    out = []
    for (first, second, ben, kind, goal, w), vals in acc.items():
        out.append({"x": first[0], "qx": first[1], "y": second[0], "qy": second[1],
                    "p": float(np.mean([v for v, _ in vals])), "n": len(vals),
                    "both_orders": len({o for _, o in vals}) == 2,
                    "beneficiary": ben, "kind": kind, "goal_good": goal, "wording": w})
    return out


def _sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


class UtilityFit:
    """One value per option (log shape), one slope b; cross-entropy against the cell probabilities."""

    def __init__(self, recs: list[dict]):
        self.keys = sorted({r["x"] for r in recs} | {r["y"] for r in recs})
        self.idx = {k: i for i, k in enumerate(self.keys)}
        xi = np.array([self.idx[r["x"]] for r in recs])
        yi = np.array([self.idx[r["y"]] for r in recs])
        d = np.log(np.array([r["qx"] for r in recs], float)) - np.log(np.array([r["qy"] for r in recs], float))
        p = np.clip(np.array([r["p"] for r in recs]), EPS, 1 - EPS)
        K = len(self.keys)

        def loss(th):
            u = np.concatenate([[0.0], th[:K - 1]])
            b = th[K - 1]
            s = _sig(b * d + u[xi] - u[yi])
            L = -np.sum(p * np.log(s + 1e-12) + (1 - p) * np.log(1 - s + 1e-12))
            g = s - p
            gu = np.zeros(K)
            np.add.at(gu, xi, g)
            np.add.at(gu, yi, -g)
            return L, np.concatenate([gu[1:], [np.sum(g * d)]])

        th0 = np.zeros(K)
        th0[K - 1] = 1.0
        res = minimize(loss, th0, jac=True, method="L-BFGS-B")
        self.u = np.concatenate([[0.0], res.x[:K - 1]])
        self.b = float(res.x[K - 1])
        self.n = len(recs)

    def predict(self, recs):
        xi = np.array([self.idx[r["x"]] for r in recs])
        yi = np.array([self.idx[r["y"]] for r in recs])
        d = np.log(np.array([r["qx"] for r in recs], float)) - np.log(np.array([r["qy"] for r in recs], float))
        return _sig(self.b * d + self.u[xi] - self.u[yi])

    def rate(self, x: str, y: str) -> float:
        """How many units of y one unit of x is worth (inf/0 if the slope is gone)."""
        if x not in self.idx or y not in self.idx:
            return float("nan")
        if self.b <= 1e-6:      # quantity no longer matters: x wins (inf) or loses (0) whatever the numbers
            diff = self.u[self.idx[x]] - self.u[self.idx[y]]
            return float("inf") if diff > 0 else (0.0 if diff < 0 else float("nan"))
        with np.errstate(over="ignore"):
            return float(np.exp((self.u[self.idx[x]] - self.u[self.idx[y]]) / self.b))


def pair_recs(recs, x, y):
    return [r for r in recs if {r["x"], r["y"]} == {x, y}]


def pair_rate(recs: list[dict], x: str, y: str) -> float:
    """Exchange rate of x in units of y, fitted from the x–y cells alone."""
    sub = pair_recs(recs, x, y)
    return UtilityFit(sub).rate(x, y) if sub else float("nan")


def miss_factor(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return float("inf")
    return float(np.exp(abs(np.log(a) - np.log(b))))


def heldout_rate_misses(recs: list[dict], pairs: list[tuple[str, str]]) -> dict:
    """Criterion 2: for each pair, predict its exchange rate from one value per good fitted on
    all OTHER pairs, and compare with the rate fitted from the pair's own cells."""
    out = {}
    for x, y in pairs:
        train = [r for r in recs if {r["x"], r["y"]} != {x, y}]
        keys = {r["x"] for r in train} | {r["y"] for r in train}
        pred = UtilityFit(train).rate(x, y) if x in keys and y in keys else float("nan")
        obs = pair_rate(recs, x, y)
        out[f"{x} vs {y}"] = {"predicted": pred, "own": obs, "miss": miss_factor(pred, obs)}
    return out


def cv_correlation(recs: list[dict]) -> float:
    """Leave one pair of goods out at a time; correlation of predicted and actual cell probabilities."""
    pairs = sorted({tuple(sorted((r["x"], r["y"]))) for r in recs})
    preds, obs = [], []
    for pr in pairs:
        held = pair_recs(recs, *pr)
        train = [r for r in recs if tuple(sorted((r["x"], r["y"]))) != pr]
        keys = {r["x"] for r in train} | {r["y"] for r in train}
        if pr[0] not in keys or pr[1] not in keys:
            continue
        preds += list(UtilityFit(train).predict(held))
        obs += [r["p"] for r in held]
    if len(preds) < 3 or np.std(preds) == 0 or np.std(obs) == 0:
        return float("nan")
    return float(np.corrcoef(preds, obs)[0, 1])


def transitivity_factor(recs: list[dict], trio: list[str]) -> float:
    """A-to-C rate (from A–C cells alone) against A-to-B x B-to-C (each from its own cells)."""
    a, b, c = trio
    r_ab, r_bc, r_ac = pair_rate(recs, a, b), pair_rate(recs, b, c), pair_rate(recs, a, c)
    if not all(np.isfinite([r_ab, r_bc, r_ac])) or min(r_ab, r_bc, r_ac) <= 0:
        return float("inf")
    return float(np.exp(abs(np.log(r_ac) - np.log(r_ab) - np.log(r_bc))))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def quantity_slope(recs: list[dict], target: str):
    """How strongly the chance of choosing `target` rises with its own number (log scale).
    OLS of logit P(target) on log q_target, with log q_partner and a fixed effect per partner.
    Returns (slope, SE, number of cells)."""
    rows = []
    for r in recs:
        if target not in (r["x"], r["y"]):
            continue
        first = r["x"] == target
        q_t, q_o, other = (r["qx"], r["qy"], r["y"]) if first else (r["qy"], r["qx"], r["x"])
        rows.append((np.log(q_t), np.log(q_o), other, float(logit(r["p"] if first else 1 - r["p"]))))
    if len(rows) < 5:
        return float("nan"), float("nan"), len(rows)
    partners = sorted({r[2] for r in rows})
    cols = [np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.ones(len(rows))]
    cols += [np.array([1.0 if r[2] == p else 0.0 for r in rows]) for p in partners[1:]]
    X = np.column_stack(cols)
    y = np.array([r[3] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
    return float(beta[0]), float(np.sqrt(max(cov[0, 0], 0))), len(rows)


def cell_p(recs: list[dict], x: str, qx: int, y: str, qy: int) -> float:
    """P(x chosen) in the cell with x at qx and y at qy (nan if that cell does not exist)."""
    for r in recs:
        if r["x"] == x and r["y"] == y and r["qx"] == qx and r["qy"] == qy:
            return r["p"]
        if r["x"] == y and r["y"] == x and r["qx"] == qy and r["qy"] == qx:
            return 1 - r["p"]
    return float("nan")
