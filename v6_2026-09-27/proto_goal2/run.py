"""Goal-2 prototype experiment: F (faithful) / C (counterfactual constants) / L (late-onset instability) worlds.
Execution-graded open schedule design; score normalized to world physics (not to an author key)."""
import json, sys, time, os
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world import (sample_world, graded_loss, Lab, sched_const, sched_cosine, sched_wsd, final_loss, diverge_step)
from opt import optimize_schedule, fit_sibling
from policies import P0_const, P1_cosine, P2_wsd, P3_grid, P4_model, P5_sawtooth, _safe_wu

T = 10000
BUDGET = int(2.5 * T)
KINDS = ('F', 'C', 'L')
SEEDS = (1, 2, 3, 4)
LAB_SEEDS = (7, 8)
GATE = 0.25


def anchors(c, T):
    grid = np.linspace(0.05, 1.2, 47) * c['th_inf']
    best = min(((graded_loss(c, sched_const(T, pk, 0.02 * T)), pk) for pk in grid), key=lambda z: z[0])
    base = sched_const(T, best[1], 0.02 * T)
    opt, _ = optimize_schedule(c, T, margin=0.94, iters=400)
    return base, opt


def validated_set(c, T, n=24, seed=0):
    r = np.random.default_rng(seed)
    out = []
    edge = dict(th_inf=c['th_inf'], th0=c['th0'], Sc=c['Sc'])
    while len(out) < n:
        H = int(r.choice([T // 4, T // 2, T]))
        pk = float(r.uniform(0.3, 0.9) * c['th_inf'])
        w = _safe_wu(pk, edge, 0.9, H)
        if r.random() < 0.5:
            eta = sched_cosine(H, pk, w, float(r.uniform(0, 0.2)))
        else:
            eta = sched_wsd(H, pk, w, float(r.uniform(0.05, 0.8)))
        if diverge_step(c, eta, 1.0) is None:
            out.append(eta)
    return out


def score(c, sib, anc, eta):
    base, opt = anc['base'], anc['opt']
    Lb, Lo = anc['Lb'], anc['Lo']
    L = graded_loss(c, eta, 'mpl')
    s = (Lb - L) / (Lb - Lo)
    Lsb, Lso = anc['Lsb'], anc['Lso']
    Ls = graded_loss(sib, eta, 'mom')
    ss = (Lsb - Ls) / (Lsb - Lso)
    flag = bool(abs(s - ss) > GATE)
    unphysical = bool(L < c['L0'])            # physical bound: below the irreducible loss
    return dict(L=L, s=s, L_sib=Ls, s_sib=ss, flag=flag, unphysical=unphysical,
                s_final=(min(s, ss) if (flag or unphysical) else s), diverged=diverge_step(c, eta, 1.0))


def one_world(args):
    kind, seed = args
    t0 = time.time()
    c = sample_world(kind, seed, T)
    out = dict(kind=kind, seed=seed, consts={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                           for k, v in c.items()})
    # sibling (ensemble gate), fitted only on the validated (monotone) domain
    fitset = validated_set(c, T, 24, seed=seed)
    sib = fit_sibling(c, fitset)
    held = validated_set(c, T, 40, seed=seed + 1000)
    anc = {}
    for H in (T, 2 * T):
        base, opt = anchors(c, H)
        a = dict(base=base, opt=opt, Lb=graded_loss(c, base), Lo=graded_loss(c, opt),
                 Lsb=graded_loss(sib, base, 'mom'), Lso=graded_loss(sib, opt, 'mom'))
        anc[H] = a
    gap = anc[T]['Lb'] - anc[T]['Lo']
    dis = [abs(final_loss(c, e, 1.0, 'mpl') - final_loss(sib, e, 1.0, 'mom')) / gap for e in held]
    out['sibling'] = dict(lam=sib['lam'], eps=sib['eps'], fit_rmse=sib['sib_rmse'],
                          heldout_disagree_over_gap_median=float(np.median(dis)),
                          heldout_disagree_over_gap_max=float(np.max(dis)))
    out['anchors'] = {str(H): dict(L_base=anc[H]['Lb'], L_opt=anc[H]['Lo'], gap=anc[H]['Lb'] - anc[H]['Lo'],
                                   opt_peak=float(anc[H]['opt'].max()),
                                   opt_eta_at=[float(anc[H]['opt'][int(f * H) - 1]) for f in (0.25, 0.5, 0.75, 0.9, 1.0)])
                      for H in (T, 2 * T)}
    res = {}
    runs = [('P0_const', P0_const, None), ('P1_cosine', P1_cosine, None), ('P2_wsd', P2_wsd, None)]
    for name, fn, _ in runs:
        r = fn(None, T)
        res[name] = [dict(T=score(c, sib, anc[T], r['eta']), T2=score(c, sib, anc[2 * T], r['gen'](2 * T)), info={})]
    for name, fn in (('P3_grid', lambda lab: P3_grid(lab, T)),
                     ('P4_model', lambda lab: P4_model(lab, T, validate=False)),
                     ('P4v_model_validated', lambda lab: P4_model(lab, T, validate=True))):
        res[name] = []
        for ls in LAB_SEEDS:
            lab = Lab(c, BUDGET, seed=ls * 100 + seed)
            r = fn(lab)
            info = {k: v for k, v in r['info'].items()}
            res[name].append(dict(T=score(c, sib, anc[T], r['eta']), T2=score(c, sib, anc[2 * T], r['gen'](2 * T)),
                                  info=json.loads(json.dumps(info, default=float))))
    # red team: sawtooth on top of the world optimum, tuned on the primary world
    saw, sp = P5_sawtooth(anc[T]['opt'], c, lambda cc, e, k: graded_loss(cc, e, k))
    res['P5_redteam_sawtooth'] = [dict(T=score(c, sib, anc[T], saw), info=sp)]
    # small-amplitude sweep: does an exploit exist that stays physically plausible (L >= L0)?
    sweep = []
    opt = anc[T]['opt']; t_pk = int(np.argmax(opt)); n = len(opt)
    for a in (0.005, 0.01, 0.02, 0.05, 0.1):
        for P in (2, 4, 10, 50, 200):
            sq = ((np.arange(n) // max(P // 2, 1)) % 2).astype(float); sq[:t_pk + 1] = 0.0
            sc = score(c, sib, anc[T], opt * (1 - a * sq))
            sweep.append(dict(a=a, P=P, s=sc['s'], s_sib=sc['s_sib'], L_minus_L0=sc['L'] - c['L0'], flag=sc['flag']))
    out['sawtooth_sweep'] = sweep
    res['OPT_world'] = [dict(T=score(c, sib, anc[T], anc[T]['opt']), T2=score(c, sib, anc[2 * T], anc[2 * T]['opt']))]
    out['results'] = res
    # planted builder bug demo (|drop|): invisible on the monotone validated domain
    if kind == 'F':
        cb = dict(c, absbug=True)
        sibb = fit_sibling(cb, fitset)
        a = anc[T]
        ab = dict(a, Lb=graded_loss(cb, a['base']), Lo=graded_loss(cb, a['opt']),
                  Lsb=graded_loss(sibb, a['base'], 'mom'), Lso=graded_loss(sibb, a['opt'], 'mom'))
        sawb, spb = P5_sawtooth(a['opt'], cb, lambda cc, e, k: graded_loss(cc, e, k))
        disb = [abs(final_loss(cb, e, 1.0, 'mpl') - final_loss(sibb, e, 1.0, 'mom')) / gap for e in held]
        out['bug_demo'] = dict(sawtooth=spb, score=score(cb, sibb, ab, sawb),
                               heldout_disagree_over_gap_max=float(np.max(disb)),
                               sibling_same_as_clean=bool(abs(sibb['sib_rmse'] - sib['sib_rmse']) < 1e-9))
    out['secs'] = time.time() - t0
    return out


def _job(js):
    o = one_world(js)
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(d, exist_ok=True)
    json.dump(o, open(os.path.join(d, '%s%d.json' % js), 'w'), indent=1, default=float)
    return '%s%d %.0fs' % (js[0], js[1], o['secs'])


if __name__ == '__main__':
    # usage: python3 run.py F1 F2 C1 L3   (each world ~80 s single-core; run <=4 per call)
    jobs = [(a[0], int(a[1:])) for a in sys.argv[1:]] or [(k, s) for k in KINDS for s in SEEDS]
    with Pool(min(4, len(jobs))) as p:
        for r in p.imap_unordered(_job, jobs):
            print(r, flush=True)
