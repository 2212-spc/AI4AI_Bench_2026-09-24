"""Baseline: Tissue et al. 2024 'Scaling law with learning-rate annealing'
    L(t) = L0 + A * S1(t)^(-alpha) - C * S2(t),
    S1 = sum lr,  S2 = sum_i m_i,  m_i = lam * m_{i-1} + (lr_{i-1} - lr_i)   (annealing momentum)."""
import numpy as np
from scipy.optimize import least_squares


def _S(lrs, lam):
    S1 = np.cumsum(lrs)
    d = np.concatenate([[0.0], lrs[:-1] - lrs[1:]])
    m = np.zeros_like(lrs); acc = 0.0
    for i, di in enumerate(d):
        acc = lam * acc + di; m[i] = acc
    return S1, np.cumsum(m)


def fit_predict(train, queries):
    best = None
    for lam in [0.99, 0.995, 0.999, 0.9995]:
        X1, X2, Y = [], [], []
        for c in train.values():
            S1, S2 = _S(c["lrs"], lam)
            X1.append(S1[c["steps"] - 1]); X2.append(S2[c["steps"] - 1]); Y.append(c["loss"])
        x1, x2, y = map(np.concatenate, (X1, X2, Y))

        def f(p, x1, x2):
            L0, logA, alpha, C = p
            return L0 + np.exp(logA) * np.maximum(x1, 1e-9) ** (-alpha) - C * x2

        p0 = [y.min() - 0.1, np.log(max(y.max() - y.min(), 1e-3)), 0.5, 0.0]
        r = least_squares(lambda p: f(p, x1, x2) - y, p0, loss="soft_l1", f_scale=0.02, max_nfev=5000)
        if best is None or r.cost < best[0]:
            best = (r.cost, lam, r.x, f)
    _, lam, p, f = best
    out = {}
    for k, lrs in queries.items():
        S1, S2 = _S(lrs, lam)
        out[k] = float(f(p, S1[-1:], S2[-1:])[0])
    return out
