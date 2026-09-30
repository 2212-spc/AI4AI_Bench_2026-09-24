"""Scripted policies (no model calls). Each returns dict(eta=<schedule at T>, gen=<callable T2 -> schedule>, info=...)."""
import numpy as np
from world import sched_const, sched_cosine, sched_wsd, _theta
from opt import optimize_schedule, fit_edge, fit_mpl

DOC = 1.0


def P0_const(lab, T):
    g = lambda T2: sched_const(T2, DOC, 0.01 * T2)
    return dict(eta=g(T), gen=g, info={})


def P1_cosine(lab, T):
    g = lambda T2: sched_cosine(T2, DOC, 0.01 * T2, 0.1)
    return dict(eta=g(T), gen=g, info={})


def P2_wsd(lab, T):
    g = lambda T2: sched_wsd(T2, DOC, 0.01 * T2, 0.2)
    return dict(eta=g(T), gen=g, info={})


def P3_grid(lab, T):
    """Author-style reference: grid-search WSD at T/10, transfer fractions to T."""
    Ts = T // 10
    res = []
    for pk in (0.6, 0.8, 1.0, 1.25, 1.5, 1.8):
        for d in (0.1, 0.2, 0.4):
            ts, obs, ds = lab.run(sched_wsd(Ts, pk, 0.02 * Ts, d))
            if ds is None:
                res.append((float(np.mean(obs[-2:])), pk, d))
    res.sort()
    _, pk, d = res[0]
    g = lambda T2: sched_wsd(T2, pk, 0.02 * T2, d)
    return dict(eta=g(T), gen=g, info=dict(peak=pk, decay=d, used=lab.used))


def _safe_wu(peak, edge, margin, T):
    for wf in (0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.25):
        w = max(int(wf * T), 5)
        eta = sched_const(w + 1, peak, w)
        c = dict(edge, tc=None, psi=0.0)
        if np.all(eta < margin * _theta(c, eta, 1.0)):
            return w
    return int(0.25 * T)


def P4_model(lab, T, validate=False, fit_len=2000):
    """Probe the edge with 3 ramps, fit the MPL family on 3 short diverse runs, optimize a free monotone
    schedule under the fitted model. validate=True: spend remaining budget on full-horizon pilots."""
    obs = []
    for D in (20, 100, 400, 1500):
        eta = 4.0 * np.arange(1, D + 1) / D
        ts, o, ds = lab.run(eta)
        if ds is not None:
            obs.append((float(eta[ds - 1]), float(np.sum(eta[:ds - 1]))))
    edge = fit_edge(obs)
    runs = []
    for frac, shape in ((0.45, 'const'), (0.7, 'wsd'), (0.85, 'cos')):
        pk = frac * edge['th_inf']
        w = _safe_wu(pk, edge, 0.85, fit_len)
        eta = {'const': sched_const(fit_len, pk, w), 'wsd': sched_wsd(fit_len, pk, w, 0.4),
               'cos': sched_cosine(fit_len, pk, w, 0.1)}[shape]
        ts, o, ds = lab.run(eta)
        if len(ts):
            runs.append((eta if ds is None else eta[:ds - 1], ts, o))
    ch = fit_mpl(runs, edge, starts=6, iters=700)
    plan, _ = optimize_schedule(ch, T, margin=0.9, iters=300)
    info = dict(edge_fit=edge, fit_rmse=ch['fit_rmse'], fitted={k: ch[k] for k in ('L0', 'A', 'alpha', 'B', 'C', 'beta', 'gamma', 'p')},
                pilots=[])
    caps = []  # list of (t_from, cap_value) in absolute steps

    def capfun(T2):
        if not caps:
            return None
        cap = np.full(T2, np.inf)
        t = np.arange(1, T2 + 1)
        for t0, v in caps:
            cap = np.where(t >= t0, np.minimum(cap, v), cap)
        return cap

    if validate:
        for _ in range(2):
            if lab.left() < T:
                break
            ts, o, ds = lab.run(plan)
            info['pilots'].append(dict(diverged=ds, final_obs=float(o[-1]) if len(o) else None))
            if ds is None:
                break
            caps.append((int(0.8 * ds), 0.85 * float(plan[ds - 1])))
            plan, _ = optimize_schedule(ch, T, margin=0.9, cap=capfun(T), iters=300)
    info['caps'] = caps
    info['used'] = lab.used
    g = lambda T2: plan if T2 == T else optimize_schedule(ch, T2, margin=0.9, cap=capfun(T2), iters=300)[0]
    return dict(eta=plan, gen=g, info=info)


def P5_sawtooth(base_eta, c, grade, kind='mpl'):
    """Red team with world access: add a downward sawtooth after the peak; pick the best (a, P) on the primary world."""
    t_pk = int(np.argmax(base_eta))
    best = (grade(c, base_eta, kind), 0.0, 0)
    n = len(base_eta)
    for a in (0.05, 0.1, 0.2, 0.4):
        for P in (2, 4, 10, 50, 200):
            sq = ((np.arange(n) // max(P // 2, 1)) % 2).astype(float)
            sq[:t_pk + 1] = 0.0
            eta = base_eta * (1 - a * sq)
            L = grade(c, eta, kind)
            if L < best[0]:
                best = (L, a, P)
    L, a, P = best
    sq = ((np.arange(n) // max(P // 2, 1)) % 2).astype(float)
    sq[:t_pk + 1] = 0.0
    return base_eta * (1 - a * sq), dict(a=a, P=P)
