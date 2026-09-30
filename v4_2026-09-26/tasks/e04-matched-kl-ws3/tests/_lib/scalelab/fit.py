"""numpy-only nonlinear least squares: Levenberg-Marquardt with numerical Jacobian, box constraints via
a logistic transform, and multi-start.  Used by oracles and rivals (never shipped to the agent)."""
import numpy as np


def _lm(resid, z0, max_iter=150, tol=1e-10):
    z = np.array(z0, float)
    r = resid(z)
    if not np.all(np.isfinite(r)):
        return z, np.inf
    cost = float(r @ r); lam = 1e-3
    for _ in range(max_iter):
        J = np.empty((r.size, z.size))
        for j in range(z.size):
            h = 1e-6 * max(1.0, abs(z[j]))
            zz = z.copy(); zz[j] += h
            rj = resid(zz)
            if not np.all(np.isfinite(rj)):
                zz[j] = z[j] - h; rj = resid(zz); J[:, j] = (r - rj) / h
            else:
                J[:, j] = (rj - r) / h
        A = J.T @ J; g = J.T @ r
        improved = False
        while lam < 1e12:
            try:
                dz = -np.linalg.solve(A + lam * (np.diag(np.diag(A)) + 1e-9 * np.eye(z.size)), g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            zn = z + dz; rn = resid(zn)
            if np.all(np.isfinite(rn)):
                cn = float(rn @ rn)
                if cn < cost:
                    rel = (cost - cn) / max(cost, 1e-300)
                    z, r, cost = zn, rn, cn; lam = max(lam / 3, 1e-12); improved = True
                    break
            lam *= 4
        if not improved or rel < tol:
            break
    return z, cost


class Box:
    """Map named bounded parameters <-> unconstrained vector.  log=True parameters are bounded in log space."""
    def __init__(self, spec):
        self.names = list(spec)
        self.lo = []; self.hi = []; self.log = []
        for k in self.names:
            lo, hi, lg = spec[k]
            self.log.append(lg)
            self.lo.append(np.log(lo) if lg else lo); self.hi.append(np.log(hi) if lg else hi)
        self.lo = np.array(self.lo); self.hi = np.array(self.hi); self.log = np.array(self.log)

    def to_params(self, z):
        s = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        v = self.lo + (self.hi - self.lo) * s
        return {k: (float(np.exp(x)) if lg else float(x)) for k, x, lg in zip(self.names, v, self.log)}

    def random_z(self, rng):
        return rng.uniform(-2.0, 2.0, size=len(self.names))

    def from_params(self, d):
        v = np.array([np.log(d[k]) if lg else d[k] for k, lg in zip(self.names, self.log)])
        s = np.clip((v - self.lo) / (self.hi - self.lo), 1e-6, 1 - 1e-6)
        return np.log(s / (1 - s))


def fit(model, box, fixed, y, w, rng, n_starts=12, init=None):
    """model(params_dict) -> predictions array aligned with y.  w: weights (1/sigma).  Returns (params, chi2)."""
    y = np.asarray(y, float); w = np.asarray(w, float)

    def resid(z):
        p = dict(fixed); p.update(box.to_params(z))
        with np.errstate(all="ignore"):
            return (np.asarray(model(p), float) - y) * w
    best = (None, np.inf)
    starts = [box.from_params(init)] if init else []
    starts += [box.random_z(rng) for _ in range(n_starts)]
    for z0 in starts:
        z, c = _lm(resid, z0)
        if c < best[1]:
            best = (z, c)
    p = dict(fixed); p.update(box.to_params(best[0]))
    return p, best[1]
