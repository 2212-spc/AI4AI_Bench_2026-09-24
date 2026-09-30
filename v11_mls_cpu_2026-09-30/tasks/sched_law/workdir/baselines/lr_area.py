"""Baseline: loss as a power law in cumulative learning-rate area  L = E + A * (S1 + s0)^(-alpha).
(The 'lr-area / effective time' rule used by many schedule-transfer papers.)"""
import numpy as np
from scipy.optimize import least_squares


def _S1(lrs):
    return np.cumsum(lrs)


def fit_predict(train, queries):
    xs, ys = [], []
    for c in train.values():
        S1 = _S1(c["lrs"])
        xs.append(S1[c["steps"] - 1]); ys.append(c["loss"])
    x, y = np.concatenate(xs), np.concatenate(ys)

    def f(p, x):
        E, logA, alpha, logs0 = p
        return E + np.exp(logA) * (x + np.exp(logs0)) ** (-alpha)

    p0 = [y.min() - 0.1, np.log(max(y.max() - y.min(), 1e-3)), 0.5, 0.0]
    r = least_squares(lambda p: f(p, x) - y, p0, loss="soft_l1", f_scale=0.02, max_nfev=5000)
    return {k: float(f(r.x, _S1(lrs)[-1:])[0]) for k, lrs in queries.items()}
