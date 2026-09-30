"""Schedule optimizer (monotone-after-warmup domain), Nelder-Mead, fitting utilities. numpy only."""
import numpy as np
from world import (_theta, loss_at, sched_wsd, _eff_area, _mom_series, EPS_ETA)

KN = np.array([0, .2, .4, .55, .65, .72, .78, .83, .87, .9, .93, .95, .97, .985, 1.0])
NZ = len(KN) - 1


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def build(params, T):
    lw, lpk, lk = params[:3]
    z = np.asarray(params[3:])
    w = float(np.clip(np.exp(lw), 5, 0.3 * T))
    pk = float(np.exp(lpk))
    kap = float(np.clip(np.exp(lk), 0.3, 3.0))
    q = np.concatenate([[1.0], np.cumprod(_sig(z))])
    t = np.arange(1, T + 1, dtype=float)
    f = np.clip((t - w) / max(T - w, 1.0), 0, 1)
    body = pk * np.interp(f, KN, q)
    ramp = pk * (np.minimum(t, w) / w) ** kap
    return np.where(t <= w, ramp, body)


def params_from_wsd(T, peak, wu, decay_frac, final=1e-3):
    w = max(wu, 5)
    fd = np.clip((T * (1 - decay_frac) - w) / (T - w), 0, 1)
    q = np.where(KN <= fd, 1.0, 1.0 - (1 - final) * (KN - fd) / max(1 - fd, 1e-9))
    q = np.maximum(q, final)
    ratio = np.clip(q[1:] / q[:-1], 1e-4, 1 - 1e-6)
    z = np.log(ratio / (1 - ratio))
    return np.concatenate([[np.log(w), np.log(peak), 0.0], z])


def objective(c, T, margin, cap=None, kind='mpl'):
    def f(params):
        eta = build(params, T)
        th = _theta(c, eta, margin)
        v = np.maximum(eta / th - 1.0, 0.0)
        pen = 1e3 * float(np.sum(v * v)) + 10.0 * float(np.sum(v > 0)) / T
        if cap is not None:
            vc = np.maximum(eta / cap - 1.0, 0.0)
            pen += 1e3 * float(np.sum(vc * vc)) + 10.0 * float(np.sum(vc > 0)) / T
        return loss_at(c, eta, None, 1.0, kind) + pen
    return f


def adam_fd(f, x0, iters=500, lr=0.03, h=1e-3, seed=0):
    x = np.array(x0, float)
    m = np.zeros_like(x); v = np.zeros_like(x)
    best, fb = x.copy(), f(x)
    for it in range(1, iters + 1):
        f0 = f(x)
        if f0 < fb:
            best, fb = x.copy(), f0
        g = np.empty_like(x)
        for i in range(len(x)):
            xi = x.copy(); xi[i] += h
            g[i] = (f(xi) - f0) / h
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** it); vh = v / (1 - 0.999 ** it)
        step = lr * mh / (np.sqrt(vh) + 1e-12)
        x = x - step
        x[3:] = np.clip(x[3:], -12, 14)
    f0 = f(x)
    if f0 < fb:
        best, fb = x.copy(), f0
    return best, fb


def optimize_schedule(c, T, margin=0.94, cap=None, iters=500, kind='mpl', init_grid=True):
    """Coarse WSD grid in the model -> Adam(FD) refinement of a free monotone profile."""
    f = objective(c, T, margin, cap, kind)
    cands = []
    for pk in np.linspace(0.3, 1.0, 8) * c['th_inf'] * margin:
        for wu in (0.005, 0.02, 0.06, 0.15):
            for d in (0.1, 0.2, 0.35, 0.5, 0.7, 0.95):
                p0 = params_from_wsd(T, pk, wu * T, d)
                cands.append((f(p0), p0))
    cands.sort(key=lambda z: z[0])
    best = None
    for fv, p0 in cands[:2]:
        x, fx = adam_fd(f, p0, iters=iters)
        if best is None or fx < best[1]:
            best = (x, fx)
    return build(best[0], T), best


# ------------------------------------------------------------------ Nelder-Mead
def nelder_mead(f, x0, step, iters=800, tol=1e-10):
    n = len(x0)
    pts = [np.array(x0, float)]
    for i in range(n):
        p = np.array(x0, float); p[i] += step[i]; pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals); pts = [pts[i] for i in order]; vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        cen = np.mean(pts[:-1], axis=0)
        xr = cen + (cen - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = cen + 2 * (cen - pts[-1]); fe = f(xe)
            if fe < fr:
                pts[-1], vals[-1] = xe, fe
            else:
                pts[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0]); vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


# ------------------------------------------------------------------ edge fit from ramp probes
def fit_edge(obs):
    """obs: list of (eta_div, area_before_div) from ramps sorted fast->slow.
    Model: eta_d = th_inf - (th_inf - th0) * exp(-A/Sc).  Robust version: th_inf is pinned to the slowest ramp's
    divergence LR (a slow ramp reaches the plateau), th0 to the fastest; only Sc is fitted (1-D grid)."""
    obs = np.array(sorted(obs, key=lambda z: z[1]), float)
    if len(obs) == 0:
        return dict(th_inf=1.0, th0=1.0, Sc=10.0)
    th_inf = float(obs[-1, 0]); th0 = float(min(obs[0, 0], th_inf))
    best = None
    for Sc in np.exp(np.linspace(np.log(1), np.log(3000), 200)):
        pred = th_inf - (th_inf - th0) * np.exp(-obs[:, 1] / Sc)
        r = float(np.sum((pred - obs[:, 0]) ** 2))
        if best is None or r < best[0]:
            best = (r, Sc)
    return dict(th_inf=th_inf, th0=th0, Sc=float(best[1]))


# ------------------------------------------------------------------ MPL-family curve fit (variable projection)
def _mpl_features(c, eta, ts):
    """For each eval t: (S_eff(t), peak(t) - sum d G) given nonlinear params in c."""
    u = _eff_area(c, eta, 1.0)
    cu = c['s0'] + np.cumsum(u)
    rmax = np.maximum.accumulate(eta)
    t_pk_full = int(np.argmax(eta))
    out_S, out_P = [], []
    for t in ts:
        e = eta[:t]
        t_pk = int(np.argmax(e))
        P = float(e[t_pk])
        if t > t_pk + 1:
            d = e[t_pk:-1] - e[t_pk + 1:]
            rc = np.cumsum(e[::-1])[::-1]
            x = np.maximum(e[t_pk + 1:], EPS_ETA) ** (-c['gamma']) * rc[t_pk + 1:]
            G = 1.0 - (c['C'] * x + 1.0) ** (-c['beta'])
            P -= float(np.dot(d, G))
        out_S.append(cu[t - 1]); out_P.append(P)
    return np.array(out_S), np.array(out_P)


def fit_mpl(runs, edge, s0=10.0, starts=4, iters=500, seed=0):
    """runs: list of (eta, ts, obs). Fits alpha,C,beta,gamma,p by NM; L0,A,B by LS."""
    rng = np.random.default_rng(seed)
    base = dict(s0=s0, tc=None, psi=0.0, **edge)
    ys = np.concatenate([r[2] for r in runs])

    def unpack(phi):
        a, lC, b, g, lp = phi
        return dict(base, alpha=float(np.clip(a, 0.05, 1.5)), C=float(np.exp(np.clip(lC, -12, 3))),
                    beta=float(np.clip(b, 0.05, 2.0)), gamma=float(np.clip(g, 0.0, 1.5)),
                    p=float(np.clip(np.exp(lp), 0.5, 12.0)))

    def solve(phi):
        c = unpack(phi)
        F = []
        for eta, ts, obs in runs:
            S, P = _mpl_features(c, eta, ts)
            F.append(np.stack([np.ones(len(ts)), S ** (-c['alpha']), P], 1))
        X = np.concatenate(F)
        coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
        r = float(np.sum((X @ coef - ys) ** 2))
        if coef[1] <= 0 or coef[2] <= 0:
            r += 1.0
        return r, coef, c

    best = None
    for s in range(starts):
        x0 = np.array([rng.uniform(0.3, 0.6), np.log(rng.uniform(0.002, 0.05)), rng.uniform(0.3, 0.7),
                       rng.uniform(0.3, 0.7), np.log(rng.uniform(2.0, 6.0))])
        x, fx = nelder_mead(lambda p: solve(p)[0], x0, [0.1, 0.7, 0.15, 0.15, 0.3], iters=iters)
        if best is None or fx < best[1]:
            best = (x, fx)
    r, coef, c = solve(best[0])
    c.update(L0=float(coef[0]), A=float(coef[1]), B=float(coef[2]))
    c['fit_rmse'] = float(np.sqrt(r / len(ys)))
    return c


# ------------------------------------------------------------------ sibling (momentum law) fit on the validated domain
def _mom_feats(c, eta, ts, lam, eps):
    u = _eff_area(c, eta, 1.0)
    cu = c['s0'] + np.cumsum(u)
    t_pk = int(np.argmax(eta))
    S2c = np.zeros(len(eta))
    if len(eta) > t_pk + 1:
        d = eta[t_pk:-1] - eta[t_pk + 1:]
        m = _mom_series(lam, d)
        w = np.maximum(eta[t_pk + 1:], 0.0) ** eps if eps > 0 else 1.0
        S2c[t_pk + 1:] = np.cumsum(m * w)
    rmax = np.maximum.accumulate(eta)
    idx = np.asarray(ts) - 1
    return cu[idx], rmax[idx], S2c[idx]


def fit_sibling(c, sched_set, every=250):
    """Fit momentum-law sibling to the primary world's noise-free curves on monotone schedules.
    Edge/efficiency are shared exactly. Returns sibling constants."""
    from world import curve
    data = []
    for eta in sched_set:
        ts, ls, ds = curve(c, eta, every, 1.0, 'mpl')
        data.append((eta, ts, ls))
    ys = np.concatenate([d[2] for d in data])
    best = None
    for lam in (0.99, 0.995, 0.998, 0.999, 0.9995):
        for eps in (0.0, 0.25, 0.5, 1.0):
            feats = [_mom_feats(c, eta, ts, lam, eps) for eta, ts, _ in data]
            for alpha in np.linspace(0.2, 0.8, 25):
                X = np.concatenate([np.stack([np.ones(len(S)), S ** (-alpha), P, -S2], 1) for S, P, S2 in feats])
                coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
                r = float(np.sum((X @ coef - ys) ** 2))
                if best is None or r < best[0]:
                    best = (r, lam, eps, alpha, coef)
    r, lam, eps, alpha, coef = best
    sib = dict(c)
    sib.update(L0=float(coef[0]), A=float(coef[1]), alpha=float(alpha), B=float(coef[2]), Cm=float(coef[3]),
               lam=lam, eps=eps, sib_rmse=float(np.sqrt(r / len(ys))))
    return sib
