"""Reference: the loss is a *linear response* to the schedule.
For SGD near a (locally) quadratic optimum every curvature mode h obeys
    v <- (1 - lr h)^2 v + lr^2 * noise,
so the loss at any step is  L_inf + sum_h [ a_h * B_h(t) + b_h * N_h(t) ]  with non-negative
weights and schedule-dependent basis functions
    B_h(t) = prod_{j<=t} (1 - lr_j h)^2            (bias still left in mode h)
    N_h(t) = sum_{tau<=t} lr_tau^2 prod_{tau<j<=t} (1 - lr_j h)^2   (noise accumulated in mode h).
Put h on a wide log grid, build the basis for every training schedule, fit weights by NNLS,
then evaluate the same basis on the query schedules."""
import numpy as np
from scipy.optimize import nnls


def _basis(lrs, hs, idx):
    K = len(hs)
    B = np.ones(K); N = np.zeros(K)
    outB, outN = [], []
    want = set(int(i) for i in idx)
    for t, lr in enumerate(lrs):
        g = (1 - lr * hs) ** 2
        B = g * B
        N = g * N + lr * lr
        if t in want:
            outB.append(B.copy()); outN.append(N.copy())
    return np.array(outB), np.array(outN)


def fit_predict(train, queries):
    peak = max(float(np.max(c["lrs"])) for c in train.values())
    hs = np.logspace(np.log10(1.8 / peak), np.log10(1.8 / peak) - 8, 56)
    rows, ys = [], []
    for c in train.values():
        Bm, Nm = _basis(c["lrs"], hs, c["steps"] - 1)
        rows.append(np.hstack([np.ones((len(Bm), 1)), Bm, Nm])); ys.append(c["loss"])
    X, y = np.vstack(rows), np.concatenate(ys)
    lam = 1e-4
    Xa = np.vstack([X, np.sqrt(lam) * np.hstack([np.zeros((2 * len(hs), 1)), np.eye(2 * len(hs))])])
    ya = np.concatenate([y, np.zeros(2 * len(hs))])
    scale = np.maximum(np.abs(Xa).max(0), 1e-12)
    w, _ = nnls(Xa / scale, ya, maxiter=20000)
    w = w / scale
    out = {}
    for k, lrs in queries.items():
        Bm, Nm = _basis(lrs, hs, [len(lrs) - 1])
        out[k] = float(np.hstack([[1.0], Bm[0], Nm[0]]) @ w)
    return out
