"""Pilot 11's readouts, on CPU (DESIGN.md section 3). All held-out numbers leave one value out.

- dom_direction: the difference of means between the top and bottom thirds of the training sentences by the target
  (a "mass-mean" direction), as a unit vector.
- lovo_dom: for each held-out value, learn the direction (and a straight-line fit from projection to target) on the
  other values and read the held-out sentences; returns the held-out predictions in target units, and the held-out
  projections of a second context onto the same training direction (for reading the direction in another prompt).
- lovo_ridge: ridge on standardised features (linear kernel), penalty chosen by an inner leave-one-value-out.
- partial_r: the correlation of x and y once z is regressed out of both.
"""
import numpy as np


def r(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def partial_r(x, y, z):
    Z = np.column_stack([np.ones(len(z)), z])
    ex = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ey = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return r(ex, ey)


def dom_direction(X, y):
    o = np.argsort(y)
    k = len(y) // 3
    d = X[o[-k:]].mean(0) - X[o[:k]].mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def fit_line(p, y):
    """y ~ a + c p; returns (a, c)."""
    c, a = np.polyfit(p, y, 1)
    return a, c


def lovo_dom(X, y, groups, X_other=None):
    """Held-out predictions of y from the direction, and held-out projections of X_other (same direction)."""
    pred = np.zeros(len(y))
    other = np.zeros(len(y)) if X_other is not None else None
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        v = dom_direction(X[tr], y[tr])
        a, c = fit_line(X[tr] @ v, y[tr])
        pred[te] = a + c * (X[te] @ v)
        if X_other is not None:
            other[te] = X_other[te] @ v
    return pred, other


def full_dom(X, y):
    """The direction learned on all sentences, with the price units per unit of projection (c) for steering."""
    v = dom_direction(X, y)
    a, c = fit_line(X @ v, y)
    return v, a, c


def _ridge_k(K, y, tr, te, alpha):
    yt = y[tr]
    coef = np.linalg.solve(K[np.ix_(tr, tr)] + alpha * np.eye(int(tr.sum())), yt - yt.mean())
    return yt.mean() + K[np.ix_(te, tr)] @ coef


def lovo_ridge(X, y, groups, scales=(0.01, 0.1, 1.0, 10.0)):
    """Features standardised over all sentences (no targets involved), then a linear kernel; the penalty is chosen
    by an inner leave-one-value-out on the training values."""
    A = (X - X.mean(0)) / (X.std(0) + 1e-6)
    K = A @ A.T
    d = X.shape[1]
    pred = np.zeros(len(y))
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        best, best_err = None, np.inf
        for s in scales:
            err = 0.0
            for h in np.unique(groups[tr]):
                ite, itr = tr & (groups == h), tr & (groups != h)
                err += np.sum((_ridge_k(K, y, itr, ite, s * d) - y[ite]) ** 2)
            if err < best_err:
                best, best_err = s * d, err
        pred[te] = _ridge_k(K, y, tr, te, best)
    return pred


def within_level(x, levels):
    """x with each level's mean removed."""
    out = np.array(x, dtype=float).copy()
    for L in np.unique(levels):
        out[levels == L] -= out[levels == L].mean()
    return out
