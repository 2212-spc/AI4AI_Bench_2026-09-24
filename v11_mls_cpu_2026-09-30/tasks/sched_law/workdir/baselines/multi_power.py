"""Baseline: Luo et al. 2025 'Multi-Power Law' for loss curves under arbitrary LR schedules
    L(t) = L0 + A (S1(t) + S_W)^(-alpha) - LD(t)
    LD(t) = B * sum_{k=1..t} (lr_{k-1} - lr_k) * G(lr_k^(-gamma) * S_k(t)),  S_k(t) = sum_{tau=k..t} lr_tau
    G(x) = 1 - (C x + 1)^(-beta)."""
import numpy as np
from scipy.optimize import least_squares


def _curve(p, lrs, idx, lr_floor=None):
    L0, logA, alpha, logSW, logB, logC, beta, gamma = p
    A, SW, B, C = np.exp(logA), np.exp(logSW), np.exp(logB), np.exp(logC)
    S1 = np.cumsum(lrs)
    d = np.concatenate([[0.0], lrs[:-1] - lrs[1:]])          # lr_{k-1} - lr_k
    d[: int(np.argmax(lrs)) + 1] = 0.0                      # warmup is not an annealing drop (paper: S_W)
    nz = np.nonzero(np.abs(d) > 1e-12)[0]
    out = L0 + A * (S1[idx] + SW) ** (-alpha)
    if len(nz):
        lr_k = np.maximum(lrs[nz], lr_floor if lr_floor else 1e-8)   # guard lr_k^-gamma near lr=0
        pre = S1[nz] - lrs[nz]                                  # sum_{tau<k}
        Skt = S1[idx][:, None] - pre[None, :]                   # sum_{tau=k..t}
        mask = nz[None, :] <= idx[:, None]
        G = 1 - (C * lr_k[None, :] ** (-gamma) * np.maximum(Skt, 0) + 1) ** (-beta)
        out = out - B * np.sum(np.where(mask, d[nz][None, :] * G, 0.0), axis=1)
    return out


def fit_predict(train, queries):
    peak = max(float(np.max(c["lrs"])) for c in train.values())
    fl = 0.05 * peak

    def resid(p):
        return np.concatenate([_curve(p, c["lrs"], c["steps"] - 1, fl) - c["loss"] for c in train.values()])

    ys = np.concatenate([c["loss"] for c in train.values()])
    best = None
    for gamma0 in [0.3, 0.6]:
        for logC0 in [0.0, 2.0]:
            p0 = [ys.min() - 0.1, np.log(max(ys.max() - ys.min(), 1e-3)), 0.5, 0.0, 0.0, logC0, 0.5, gamma0]
            lb = [-np.inf, -20, 0.01, -10, -20, -20, 0.01, 0.0]
            ub = [np.inf, 20, 3.0, 12, 20, 20, 5.0, 2.0]
            try:
                r = least_squares(resid, p0, bounds=(lb, ub), loss="soft_l1", f_scale=0.02, max_nfev=3000)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    return {k: float(_curve(best.x, lrs, np.array([len(lrs) - 1]), fl)[0]) for k, lrs in queries.items()}
