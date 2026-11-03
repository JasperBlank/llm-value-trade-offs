"""Pilot 12's analysis helpers, on CPU (DESIGN.md sections 2-4). Held-out numbers leave one value out."""
import numpy as np


def r(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def cubic_residual(u, R):
    """The disagreement: R minus a cubic in u, fitted over all sentences (no states involved)."""
    return R - np.polyval(np.polyfit(u, R, 3), u)


def partial_r_multi(x, y, Z):
    """Correlation of x and y once the columns of Z (and a constant) are regressed out of both."""
    A = np.column_stack([np.ones(len(x)), Z])
    ex = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ey = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return r(ex, ey)


def kernel(X):
    A = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return A @ A.T


def _ridge_k(K, y, tr, te, alpha):
    yt = y[tr]
    coef = np.linalg.solve(K[np.ix_(tr, tr)] + alpha * np.eye(int(tr.sum())), yt - yt.mean())
    return yt.mean() + K[np.ix_(te, tr)] @ coef


def lovo_ridge_K(K, y, groups, d, scales=(0.01, 0.1, 1.0, 10.0)):
    """Ridge with a precomputed linear kernel of standardised states; penalty by inner leave-one-value-out."""
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


class RidgeLOVO:
    """lovo_ridge_K with every linear map precomputed, so that a new target (a permutation) costs only products."""
    def __init__(self, K, groups, d, scales=(0.01, 0.1, 1.0, 10.0)):
        self.groups, self.alphas = groups, [s * d for s in scales]
        self.folds = []
        for g in np.unique(groups):
            te, tr = groups == g, groups != g
            inner = []
            for h in np.unique(groups[tr]):
                ite, itr = tr & (groups == h), tr & (groups != h)
                inner.append((itr, ite, [K[np.ix_(ite, itr)] @ np.linalg.inv(K[np.ix_(itr, itr)] + a * np.eye(int(itr.sum())))
                                         for a in self.alphas]))
            outer = [K[np.ix_(te, tr)] @ np.linalg.inv(K[np.ix_(tr, tr)] + a * np.eye(int(tr.sum()))) for a in self.alphas]
            self.folds.append((tr, te, inner, outer))

    def predict(self, y):
        pred = np.zeros(len(y))
        for tr, te, inner, outer in self.folds:
            errs = np.zeros(len(self.alphas))
            for itr, ite, Hs in inner:
                yi = y[itr]
                m = yi.mean()
                for k, H in enumerate(Hs):
                    errs[k] += np.sum((m + H @ (yi - m) - y[ite]) ** 2)
            k = int(np.argmin(errs))
            yt = y[tr]
            pred[te] = yt.mean() + outer[k] @ (yt - yt.mean())
        return pred


def best_layer_r(models, y):
    """Held-out r of the ridge readout at each layer (RidgeLOVO models), and the best."""
    rs = [r(m.predict(y), y) for m in models]
    return rs, int(np.argmax(rs)), float(max(rs))


def permutation_p(models, y, n_perm, seed):
    """Null: y shuffled across sentences; the whole procedure (readout at every layer, then the best) repeated."""
    rng = np.random.default_rng(seed)
    _, _, real = best_layer_r(models, y)
    null = np.array([best_layer_r(models, rng.permutation(y))[2] for _ in range(n_perm)])
    return float((1 + np.sum(null >= real)) / (1 + n_perm)), null


def pls2_lovo(X, Y, groups, k):
    """Partial least squares with k components predicting the columns of Y (z-scored on the training values),
    states standardised over all sentences; held-out predictions in Y's units."""
    A = (X - X.mean(0)) / (X.std(0) + 1e-6)
    pred = np.zeros_like(Y, dtype=float)
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        my, sy = Y[tr].mean(0), Y[tr].std(0)
        Xr, Yr = A[tr].copy(), (Y[tr] - my) / sy
        Xr -= Xr.mean(0)
        xm = A[tr].mean(0)
        W, P, Q = [], [], []
        for _ in range(k):
            w = np.linalg.svd(Xr.T @ Yr, full_matrices=False)[0][:, 0]
            t = Xr @ w
            tt = t @ t
            p_, q_ = Xr.T @ t / tt, Yr.T @ t / tt
            Xr = Xr - np.outer(t, p_)
            Yr = Yr - np.outer(t, q_)
            W.append(w), P.append(p_), Q.append(q_)
        W, P, Q = np.array(W).T, np.array(P).T, np.array(Q).T
        B = W @ np.linalg.solve(P.T @ W, Q.T)
        pred[te] = ((A[te] - xm) @ B) * sy + my
    return pred


def bootstrap_excess(Cp, Rp, Cd, Rd, n_boot, seed):
    """excess = mean(Rd) - (mean(Rp) / mean(Cp)) * mean(Cd), with a bootstrap over sentences."""
    rng = np.random.default_rng(seed)
    n = len(Cp)
    ex = lambda i: Rd[i].mean() - Rp[i].mean() / Cp[i].mean() * Cd[i].mean()
    est = float(ex(np.arange(n)))
    boots = np.array([ex(rng.integers(0, n, n)) for _ in range(n_boot)])
    return est, [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
