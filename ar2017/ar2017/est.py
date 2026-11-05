"""OLS / 2SLS with Stata conventions.

ols, ols_v : `reg, r` (HC1, k = rank) or `reg, cluster()` (G/(G-1)*(N-1)/(N-K)).
iv, iv_v   : `ivreg2, r [cluster()] partial(...)` -- no small-sample correction on V.
             First-stage F uses (N-K)/N (cluster: (G-1)/G*(N-K)/(N-1)); Hansen J uses
             the two-step efficient GMM weight from 2SLS residuals.
ar_ci      : Anderson-Rubin set by grid inversion, robust/cluster Wald in F form.

Verified against the paper (Tables 1-2, all cells) and against linearmodels.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2, f as fdist


def _resid(W, A):
    if W.shape[1] == 0:
        return A
    return A - W @ np.linalg.lstsq(W, A, rcond=None)[0]


def _meat(M, e, g):
    if g is None:
        return (M * e[:, None] ** 2).T @ M
    _, gi = np.unique(g, return_inverse=True)
    Sg = np.zeros((gi.max() + 1, M.shape[1]))
    np.add.at(Sg, gi, M * e[:, None])
    return Sg.T @ Sg


def _rank(A, rel=1e-10):
    s = np.linalg.svd(A, compute_uv=False)
    return int((s > s[0] * rel).sum())


def _prep(df, y, endog, exog, Z, cluster):
    """Drop incomplete rows, partial the exogenous block (incl. constant) out of y, X, Z."""
    endog = [endog] if isinstance(endog, str) else list(endog)
    d = df[[y] + endog + exog + Z + ([cluster] if cluster else [])].dropna()
    n = len(d)
    g = d[cluster].values if cluster else None
    W = np.column_stack([np.ones(n), d[exog].values.astype(float)]) if exog else np.ones((n, 1))
    yt = _resid(W, d[y].values.astype(float))
    Xt = _resid(W, d[endog].values.astype(float))
    Zt = _resid(W, d[Z].values.astype(float))
    U, s, _ = np.linalg.svd(Zt, full_matrices=False)       # drop instruments absorbed by W
    keep = s > s[0] * 1e-8
    Zt = U[:, keep] * s[keep]
    kfs = _rank(np.column_stack([W, Zt]))
    if g is None:
        adj = (n - kfs) / n
    else:
        G = len(np.unique(g))
        adj = (G - 1) / G * (n - kfs) / (n - 1)
    return d, n, g, W, yt, Xt, Zt, kfs, adj


def ols(df, y, X, cluster=None):
    """Returns coef Series, se Series, N."""
    b, V, n = ols_v(df, y, X, cluster)
    return b, pd.Series(np.sqrt(np.diag(V.values)), b.index), n


def ols_v(df, y, X, cluster=None):
    """Returns coef Series, vcov DataFrame, N."""
    d = df[[y] + X + ([cluster] if cluster else [])].dropna()
    Y = d[y].values.astype(float)
    M = np.column_stack([np.ones(len(d)), d[X].values.astype(float)])
    n, k = len(d), _rank(M)
    XtXi = np.linalg.pinv(M.T @ M)
    b = XtXi @ M.T @ Y
    e = Y - M @ b
    g = d[cluster].values if cluster else None
    V = XtXi @ _meat(M, e, g) @ XtXi
    if g is None:
        V *= n / (n - k)
    else:
        G = len(np.unique(g))
        V *= G / (G - 1) * (n - 1) / (n - k)
    names = ['const'] + X
    return pd.Series(b, names), pd.DataFrame(V, names, names), n


def iv(df, y, endog, exog, Z, cluster=None):
    """One endogenous regressor. Returns b, se, N, first-stage F, Hansen J p-value
    (NaN when just identified)."""
    d, n, g, W, yt, Xt, Zt, kfs, adj = _prep(df, y, endog, exog, Z, cluster)
    xt = Xt[:, 0]
    L = Zt.shape[1]
    Zi = np.linalg.pinv(Zt.T @ Zt)
    xh = Zt @ (Zi @ Zt.T @ xt)
    b = (xh @ yt) / (xh @ xt)
    e = yt - xt * b
    se = np.sqrt(float(_meat(xh[:, None], e, g)[0, 0])) / abs(xh @ xt)
    pi = Zi @ Zt.T @ xt
    u = xt - Zt @ pi
    Vp = Zi @ _meat(Zt, u, g) @ Zi
    F = float(pi @ np.linalg.pinv(Vp) @ pi) / L * adj
    if L > 1:
        Sw = np.linalg.pinv(_meat(Zt, e, g) / n)
        zx, zy = Zt.T @ xt / n, Zt.T @ yt / n
        bg = float(zx @ Sw @ zy) / float(zx @ Sw @ zx)
        gb = Zt.T @ (yt - xt * bg) / n
        pJ = 1 - chi2.cdf(float(n * gb @ Sw @ gb), L - 1)
    else:
        pJ = np.nan
    return b, se, n, F, pJ


def ar_ci(df, y, endog, exog, Z, cluster=None, grid=None, level=0.95):
    """Anderson-Rubin weak-IV-robust set by grid inversion.

    Returns (lo, hi, status). status is one of
      'bounded'   one interval strictly inside the grid
      'unbounded' accepted region touches a grid edge
      'disjoint'  accepted region has >1 piece (lo/hi are then the hull -- report with care)
      'empty'     no value accepted: AR rejects every beta. With >1 instrument this is a
                  rejection of the overidentifying restrictions, not an uninformative test.
    With L > 1 instruments the AR test is a joint test of beta = b0 AND instrument validity,
    so its sets can be shorter than Wald intervals when J is near rejection; compare with a
    just-identified version before reading significance off them.
    """
    d, n, g, W, yt, Xt, Zt, kk, adj = _prep(df, y, endog, exog, Z, cluster)
    xt = Xt[:, 0]
    L = Zt.shape[1]
    Zi = np.linalg.pinv(Zt.T @ Zt)
    crit = fdist.ppf(level, L, n - kk) if g is None else fdist.ppf(level, L, len(np.unique(g)) - 1)
    grid = np.arange(-10, 10.0001, 0.01) if grid is None else np.asarray(grid)
    acc = np.empty(len(grid), bool)
    for i, b0 in enumerate(grid):
        r = yt - b0 * xt
        pi = Zi @ Zt.T @ r
        u = r - Zt @ pi
        V = Zi @ _meat(Zt, u, g) @ Zi
        acc[i] = float(pi @ np.linalg.pinv(V) @ pi) / L * adj <= crit
    if not acc.any():
        return np.nan, np.nan, 'empty'
    lo, hi = float(np.round(grid[acc].min(), 10)), float(np.round(grid[acc].max(), 10))
    pieces = int(acc[0]) + int((np.diff(acc.astype(int)) == 1).sum())
    if acc[0] or acc[-1]:
        status = 'unbounded'
    elif pieces > 1:
        status = 'disjoint'
    else:
        status = 'bounded'
    return lo, hi, status


def iv_v(df, y, endog, exog, Z, cluster=None):
    """2SLS with several endogenous regressors. Returns coef Series, vcov DataFrame (endog),
    N, dict of Sanderson-Windmeijer first-stage F per endogenous regressor."""
    d, n, g, W, yt, Xt, Zt, kfs, adj = _prep(df, y, endog, exog, Z, cluster)
    L = Zt.shape[1]
    P = Zt @ np.linalg.pinv(Zt.T @ Zt) @ Zt.T
    Xh = P @ Xt
    A = np.linalg.pinv(Xh.T @ Xt)
    b = A @ Xh.T @ yt
    e = yt - Xt @ b
    V = A @ _meat(Xh, e, g) @ A.T
    Zi = np.linalg.pinv(Zt.T @ Zt)
    swF = {}
    for j, nm in enumerate(endog):
        others = [i for i in range(len(endog)) if i != j]
        xj = Xt[:, j]
        if others:
            Xo = Xt[:, others]
            delta = np.linalg.pinv((P @ Xo).T @ Xo) @ (P @ Xo).T @ xj
            r = xj - Xo @ delta
        else:
            r = xj
        pi = Zi @ Zt.T @ r
        u = r - Zt @ pi
        Vp = Zi @ _meat(Zt, u, g) @ Zi
        swF[nm] = float(pi @ np.linalg.pinv(Vp) @ pi) / (L - len(others)) * adj
    return pd.Series(b, endog), pd.DataFrame(V, endog, endog), n, swF
