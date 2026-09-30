"""Small numeric helpers (numpy only): Levenberg-Marquardt least squares and simplex utilities."""
import numpy as np


def lm_fit(resid, x0, lo=None, hi=None, iters=200, lam=1e-2, eps=1e-6):
    """Minimise sum(resid(x)**2) with box bounds by clipping.  Returns (x, cost)."""
    x = np.array(x0, float)
    lo = np.full_like(x, -np.inf) if lo is None else np.asarray(lo, float)
    hi = np.full_like(x, np.inf) if hi is None else np.asarray(hi, float)
    r = resid(x); c = float(r @ r)
    for _ in range(iters):
        J = np.empty((r.size, x.size))
        for k in range(x.size):
            h = eps * max(1.0, abs(x[k]))
            xp = x.copy(); xp[k] += h
            J[:, k] = (resid(xp) - r) / h
        A = J.T @ J; g = J.T @ r
        improved = False
        for _t in range(12):
            try:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            xn = np.clip(x + step, lo, hi)
            rn = resid(xn); cn = float(rn @ rn)
            if np.isfinite(cn) and cn < c:
                x, r, c = xn, rn, cn; lam = max(lam / 3, 1e-9); improved = True; break
            lam *= 5
        if not improved or np.max(np.abs(step)) < 1e-10:
            break
    return x, c


def simplex_grid(k, step):
    n = int(round(1 / step))
    def rec(k, n):
        if k == 1:
            yield (n,); return
        for a in range(n + 1):
            for rest in rec(k - 1, n - a):
                yield (a,) + rest
    return np.array(list(rec(k, n)), float) / n
