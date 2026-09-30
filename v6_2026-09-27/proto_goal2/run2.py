"""Follow-up experiment (answers three questions raised by results/*.json):
  Q1  Is the L-world horizon-transfer task (2T) solvable within budget?  -> P6 late-edge probe policy (existence proof)
  Q2  Does P4's success depend on knowing the true law family?          -> P4m fits the *sibling* (momentum) family instead
  Q3  Is the score really uncapped?                                      -> OPT_strong (margin .95, more iters) vs our anchor
Recomputes world/sibling/anchors deterministically (and checks they reproduce results/*.json), then runs the new policies.
usage: python3 run2.py F1 C1 L1 ..."""
import json, sys, time, os
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world import (sample_world, graded_loss, Lab, sched_const, sched_cosine, sched_wsd, _mom_series)
from opt import optimize_schedule, fit_sibling, fit_edge, fit_mpl, _mom_feats
from policies import _safe_wu
from run import T, BUDGET, LAB_SEEDS, anchors, validated_set, score

D = os.path.dirname(os.path.abspath(__file__))


def _probe_edge_and_fit_runs(lab, fit_len=2000):
    obs = []
    for Dr in (20, 100, 400, 1500):
        eta = 4.0 * np.arange(1, Dr + 1) / Dr
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
    return edge, runs


def fit_mom_lab(runs, edge, s0=10.0):
    """Mis-specified model: momentum-law family (Tissue et al.) fitted to noisy lab curves."""
    ys = np.concatenate([r[2] for r in runs])
    best = None
    for p in (2.0, 3.0, 4.0, 6.0):
        cf = dict(edge, p=p, s0=s0, tc=None, psi=0.0)
        for lam in (0.99, 0.995, 0.998, 0.999, 0.9995):
            for eps in (0.0, 0.25, 0.5, 1.0):
                feats = [_mom_feats(cf, eta, ts, lam, eps) for eta, ts, _ in runs]
                for alpha in np.linspace(0.2, 0.8, 25):
                    X = np.concatenate([np.stack([np.ones(len(S)), S ** (-alpha), P, -S2], 1) for S, P, S2 in feats])
                    coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
                    r = float(np.sum((X @ coef - ys) ** 2))
                    if coef[1] <= 0 or coef[2] <= 0 or coef[3] <= 0:
                        r += 1.0
                    if best is None or r < best[0]:
                        best = (r, p, lam, eps, alpha, coef)
    r, p, lam, eps, alpha, coef = best
    ch = dict(edge, p=p, s0=s0, tc=None, psi=0.0, lam=lam, eps=eps, alpha=float(alpha),
              L0=float(coef[0]), A=float(coef[1]), B=float(coef[2]), Cm=float(coef[3]),
              fit_rmse=float(np.sqrt(r / len(ys))))
    return ch


def P4m_misspecified(lab, T):
    edge, runs = _probe_edge_and_fit_runs(lab)
    ch = fit_mom_lab(runs, edge)
    plan, _ = optimize_schedule(ch, T, margin=0.9, iters=300, kind='mom')
    g = lambda T2: plan if T2 == T else optimize_schedule(ch, T2, margin=0.9, iters=300, kind='mom')[0]
    return dict(eta=plan, gen=g, info=dict(fit_rmse=ch['fit_rmse'], lam=ch['lam'], eps=ch['eps'], p=ch['p'], used=lab.used))


def P6_late_probe(lab, T, t_probe=(4000, 8000), lo_frac=0.35, ramp=300, margin=0.85):
    """P4 + two late-edge probes (hold a low LR for t0 steps, then ramp fast; the divergence step reveals the edge
    at time ~t0). Edge-vs-time extrapolated with the steepest power-law decay consistent with the evidence;
    optimize under cap(t) = margin * edge_hat(t). No knowledge of the world kind is used."""
    edge, runs = _probe_edge_and_fit_runs(lab)
    ch = fit_mpl(runs, edge, starts=6, iters=700)
    th = edge['th_inf']
    pts = []
    for t0 in t_probe:
        lo = lo_frac * th
        w = _safe_wu(lo, edge, 0.85, t0)
        eta = np.concatenate([sched_const(t0, lo, w), np.linspace(lo, 1.5 * th, ramp)])
        if lab.left() < len(eta):
            break
        ts, o, ds = lab.run(eta)
        if ds is not None:
            pts.append((ds, float(eta[ds - 1])))
    # edge-vs-time model: plateau th until tc_hat, then (t/tc_hat)^-psi_hat
    tc_hat, psi_hat = None, 0.0
    dec = [(t, v) for t, v in pts if v < 0.93 * th]
    if dec:
        flat = [t for t, v in pts if v >= 0.93 * th]
        t_last_flat = max(flat) if flat else 1500.0          # latest time the edge was seen on the plateau
        t2, v2 = dec[-1]
        if len(dec) >= 2 and dec[0][0] < t2:
            t1, v1 = dec[0]
            psi_hat = float(np.log(v1 / v2) / np.log(t2 / t1))
            tc_hat = float(t1 * (v1 / th) ** (1.0 / max(psi_hat, 1e-6)))
        else:                                               # one decayed point: steepest decay consistent with evidence
            tc_hat = float(t_last_flat)
            psi_hat = float(np.log(th / v2) / np.log(t2 / tc_hat))
        if not (psi_hat > 0 and tc_hat and tc_hat > 0):     # later edge not lower than earlier -> noise, no decay
            tc_hat, psi_hat = None, 0.0

    def capfun(H):
        if tc_hat is None:
            return None
        t = np.arange(1, H + 1, dtype=float)
        return margin * th * np.minimum(1.0, (t / tc_hat) ** (-psi_hat))

    plan, _ = optimize_schedule(ch, T, margin=0.9, cap=capfun(T), iters=300)
    g = lambda T2: plan if T2 == T else optimize_schedule(ch, T2, margin=0.9, cap=capfun(T2), iters=300)[0]
    return dict(eta=plan, gen=g, info=dict(edge_pts=pts, tc_hat=tc_hat, psi_hat=psi_hat, used=lab.used))


def one_world(args):
    kind, seed = args
    t0 = time.time()
    c = sample_world(kind, seed, T)
    fitset = validated_set(c, T, 24, seed=seed)
    sib = fit_sibling(c, fitset)
    anc = {}
    for H in (T, 2 * T):
        base, opt = anchors(c, H)
        anc[H] = dict(base=base, opt=opt, Lb=graded_loss(c, base), Lo=graded_loss(c, opt),
                      Lsb=graded_loss(sib, base, 'mom'), Lso=graded_loss(sib, opt, 'mom'))
    old = json.load(open(os.path.join(D, 'results', '%s%d.json' % (kind, seed))))
    repro = max(abs(old['anchors'][str(H)]['L_base'] - anc[H]['Lb']) + abs(old['anchors'][str(H)]['L_opt'] - anc[H]['Lo'])
                for H in (T, 2 * T))
    out = dict(kind=kind, seed=seed, anchor_repro_err=float(repro),
               true_tc=c.get('tc'), true_psi=c.get('psi'), results={})
    for name, fn in (('P4m_misspecified_family', lambda lab: P4m_misspecified(lab, T)),
                     ('P6_late_edge_probe', lambda lab: P6_late_probe(lab, T))):
        out['results'][name] = []
        for ls in LAB_SEEDS:
            lab = Lab(c, BUDGET, seed=ls * 100 + seed)
            r = fn(lab)
            out['results'][name].append(dict(T=score(c, sib, anc[T], r['eta']), T2=score(c, sib, anc[2 * T], r['gen'](2 * T)),
                                             info=json.loads(json.dumps(r['info'], default=float))))
    strong, _ = optimize_schedule(c, T, margin=0.95, iters=1000)
    out['results']['OPT_strong'] = [dict(T=score(c, sib, anc[T], strong))]
    out['secs'] = time.time() - t0
    return out


def _job(js):
    o = one_world(js)
    d = os.path.join(D, 'results2')
    os.makedirs(d, exist_ok=True)
    json.dump(o, open(os.path.join(d, '%s%d.json' % js), 'w'), indent=1, default=float)
    return '%s%d %.0fs repro=%.2e' % (js[0], js[1], o['secs'], o['anchor_repro_err'])


if __name__ == '__main__':
    jobs = [(a[0], int(a[1:])) for a in sys.argv[1:]]
    with Pool(min(4, len(jobs))) as p:
        for r in p.imap_unordered(_job, jobs):
            print(r, flush=True)
