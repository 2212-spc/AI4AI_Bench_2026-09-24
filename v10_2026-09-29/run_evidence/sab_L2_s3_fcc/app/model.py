import numpy as np, csv, collections

def erlang_a(lam, mu, c, theta, nmax=4000):
    """Steady state of M/M/c+M. Returns (P_abandon, E[wait] over all arrivals, util)."""
    # birth-death: up rate lam; down rate min(n,c)*mu + max(n-c,0)*theta
    # compute unnormalized probs via log to avoid overflow
    logp = [0.0]
    for n in range(1, nmax):
        d = min(n, c) * mu + max(n - c, 0) * theta
        logp.append(logp[-1] + np.log(lam / d))
        if n > c and logp[-1] < logp[0] - 60 and lam / d < 0.5:
            break
    logp = np.array(logp)
    p = np.exp(logp - logp.max()); p /= p.sum()
    n = np.arange(len(p))
    EQ = np.sum(p * np.maximum(n - c, 0))
    busy = np.sum(p * np.minimum(n, c))
    Pab = theta * EQ / lam
    EW = EQ / lam
    return Pab, EW, busy / c

def replicas(lam, mean_gen, cap=16):
    load = lam * mean_gen
    return int(min(cap, max(4, np.ceil(load / 0.75 - 1e-12))))

if __name__ == "__main__":
    rows = list(csv.DictReader(open('data/history.csv')))
    th = []
    for r in rows:
        ab = float(r['abandon_rate']); w = float(r['mean_queue_wait_s'])
        th.append(ab / w)
    th = np.array(th)
    print('theta est mean %.5f median %.5f sd %.5f' % (th.mean(), np.median(th), th.std()))
    # check replicas rule
    bad = 0
    for r in rows:
        lam = float(r['requests']) / 3600
        c = replicas(lam, float(r['mean_gen_time_s']))
        if c != int(float(r['replicas'])): bad += 1
    print('replica mismatches', bad, 'of', len(rows))
    # fit theta by least squares on wait & abandon
    for theta in [0.012, 0.014, 0.015, 0.016, 0.017, 0.018, 0.02]:
        errW = []; errA = []; ratio=[]
        for r in rows:
            lam = float(r['requests']) / 3600
            c = int(float(r['replicas']))
            mu = 1 / 2.535
            Pab, EW, u = erlang_a(lam, mu, c, theta)
            errW.append(float(r['mean_queue_wait_s']) / EW)
            errA.append(float(r['abandon_rate']) / Pab)
        print(theta, 'wait ratio obs/model %.4f (sd %.3f)  aband ratio %.4f (sd %.3f)' % (np.mean(errW), np.std(errW), np.mean(errA), np.std(errA)))

MA = 2.536
def mB(h):
    return 6.05 if 9 <= h <= 17 else 4.68

def load_rows():
    import glob
    out = []
    for r in csv.DictReader(open('data/history.csv')):
        out.append(dict(day=int(r['day']), hour=int(r['hour']), f=0.0, n=int(r['requests']), c=int(float(r['replicas'])),
                        ab=float(r['abandon_rate']), w=float(r['mean_queue_wait_s']), util=float(r['utilization']), src='hist'))
    for fn in sorted(glob.glob('data/abtest_*.csv')):
        for r in csv.DictReader(open(fn)):
            out.append(dict(day=int(r['day']), hour=int(r['hour']), f=float(r['fraction_B']), n=int(r['requests_A']) + int(r['requests_B']),
                            c=int(float(r['replicas'])), ab=float(r['abandon_rate']), w=float(r['mean_queue_wait_s']), util=float(r['utilization']), src=fn,
                            gA=float(r['gen_time_A_s'] or 'nan'), gB=float(r['gen_time_B_s'] or 'nan'), nA=int(r['requests_A']), nB=int(r['requests_B'])))
    return out

def mixed_queue(lam, f, mA, mB_, c, theta, Q=None):
    """Exact steady state of the shared FIFO pool with per-request random routing (prob f to B),
    exponential service per arm, exponential patience. Level-dependent QBD solved by linear level reduction.
    Returns (Pab, EW, util)."""
    muA, muB = 1.0 / mA, 1.0 / mB_
    if f <= 0: return erlang_a(lam, muA, c, theta)
    if f >= 1: return erlang_a(lam, muB, c, theta)
    if Q is None:
        excess = max(lam - c / ((1 - f) * mA + f * mB_), 0)
        Q = int(60 + 8 * excess / theta + 6 * np.sqrt(excess / theta + 1) + 3 * c)
    # level-0 states: all (a,b) with a+b<=c ; index map
    idx0 = {}
    for a in range(c + 1):
        for b in range(c + 1 - a):
            idx0[(a, b)] = len(idx0)
    n0 = len(idx0)
    P = c + 1  # phases at levels >=1: a in 0..c, b=c-a
    A1_0 = np.zeros((n0, n0)); A0_0 = np.zeros((n0, P))
    for (a, b), i in idx0.items():
        tot = 0.0
        if a + b < c:
            A1_0[i, idx0[(a + 1, b)]] += lam * (1 - f); A1_0[i, idx0[(a, b + 1)]] += lam * f; tot += lam
        else:
            A0_0[i, a] += lam; tot += lam
        if a > 0: A1_0[i, idx0[(a - 1, b)]] += a * muA; tot += a * muA
        if b > 0: A1_0[i, idx0[(a, b - 1)]] += b * muB; tot += b * muB
        A1_0[i, i] -= tot
    # level q>=1 blocks (phase a)
    A0 = np.eye(P) * lam  # up
    def down_block(q, to_level0):
        # transitions from level q to q-1
        D = np.zeros((P, n0 if to_level0 else P))
        for a in range(P):
            b = c - a
            tgt = (lambda aa, bb: idx0[(aa, bb)]) if to_level0 else (lambda aa, bb: aa)
            # A completion: head of queue starts service
            if a > 0:
                D[a, tgt(a, b)] += a * muA * (1 - f); D[a, tgt(a - 1, b + 1)] += a * muA * f
            if b > 0:
                D[a, tgt(a, b)] += b * muB * f; D[a, tgt(a + 1, b - 1)] += b * muB * (1 - f)
            D[a, tgt(a, b)] += q * theta
        return D
    def local_block(q, top):
        d = np.array([-(lam * (0 if top else 1) + a * muA + (c - a) * muB + q * theta) for a in range(P)])
        return np.diag(d)
    # backward recursion for R_q, q = Q..1 ; pi_q = pi_{q-1} R_q
    R = [None] * (Q + 2)
    R[Q + 1] = np.zeros((P, P))
    Anext_down = None
    for q in range(Q, 0, -1):
        A1q = local_block(q, top=(q == Q))
        M = A1q + (R[q + 1] @ down_block(q + 1, False) if q < Q else 0)
        Aup = A0_0 if q == 1 else A0
        R[q] = Aup @ np.linalg.inv(-M)
    M0 = A1_0 + R[1] @ down_block(1, True)
    # solve pi0 M0 = 0 with normalization: replace one equation
    Mt = M0.T.copy()
    Mt[-1, :] = 1.0
    rhs = np.zeros(n0); rhs[-1] = 1.0
    pi0 = np.linalg.solve(Mt, rhs)
    pis = [pi0]
    for q in range(1, Q + 1):
        pis.append(pis[-1] @ R[q])
    Z = sum(p.sum() for p in pis)
    pis = [p / Z for p in pis]
    EQ = sum(q * pis[q].sum() for q in range(1, Q + 1))
    busy = sum(pis[0][i] * (a + b) for (a, b), i in idx0.items()) + c * sum(pis[q].sum() for q in range(1, Q + 1))
    if pis[Q].sum() > 1e-6: print('WARN truncation', pis[Q].sum())
    return theta * EQ / lam, EQ / lam, busy / c

ETA = 0.0112
def arrivals_by_hour():
    """dict hour -> array of hourly request counts over all observed days"""
    rows = load_rows()
    out = collections.defaultdict(dict)
    for r in rows: out[r['hour']][r['day']] = r['n']
    return {h: np.array([out[h][d] for d in sorted(out[h])]) for h in out}

def hour_value(h, f, ns, rA, dB, mBh, theta, cap=16, cache={}):
    """mean score over days for hour h at fraction f (request-weighted) and at f=0."""
    tot = 0.0; tot0 = 0.0
    for n in ns:
        lam = n / 3600
        m = (1 - f) * MA + f * mBh
        c = int(min(cap, max(4, np.ceil(lam * m / 0.75 - 1e-9))))
        key = (round(lam, 6), round(f, 4), mBh, c, theta)
        if key not in cache: cache[key] = mixed_queue(lam, f, MA, mBh, c, theta)
        Pab, EW, _ = cache[key]
        tot += n * ((1 - Pab) * (rA + f * dB) - ETA * EW)
        c0 = int(min(cap, max(4, np.ceil(lam * MA / 0.75 - 1e-9))))
        key0 = (round(lam, 6), 0.0, mBh, c0, theta)
        if key0 not in cache: cache[key0] = erlang_a(lam, 1 / MA, c0, theta)
        P0, W0, _ = cache[key0]
        tot0 += n * ((1 - P0) * rA - ETA * W0)
    return tot / ns.sum(), tot0 / ns.sum()

RA = 0.6421
def dB_h(h): return (0.6772 if 9 <= h <= 17 else 0.6603) - RA
MB_PEAK, MB_OFF = 6.031, 4.671
def mB(h): return MB_PEAK if 9 <= h <= 17 else MB_OFF

def hour_value2(h, f, ns, theta, model='mixed', cap=16, rA=RA, dB=None, mBh=None, cache={}):
    dB = dB_h(h) if dB is None else dB; mBh = mB(h) if mBh is None else mBh
    tot = 0.0; tot0 = 0.0
    for n in ns:
        lam = n / 3600
        m = (1 - f) * MA + f * mBh
        c = int(min(cap, max(4, np.ceil(lam * m / 0.75 - 1e-9))))
        key = (model, round(lam, 6), round(f, 4), mBh, c, theta)
        if key not in cache:
            cache[key] = mixed_queue(lam, f, MA, mBh, c, theta) if model == 'mixed' else erlang_a(lam, 1 / m, c, theta)
        Pab, EW, _ = cache[key]
        tot += n * ((1 - Pab) * (rA + f * dB) - ETA * EW)
        c0 = int(min(cap, max(4, np.ceil(lam * MA / 0.75 - 1e-9))))
        key0 = ('e', round(lam, 6), 0.0, c0, theta)
        if key0 not in cache: cache[key0] = erlang_a(lam, 1 / MA, c0, theta)
        P0, W0, _ = cache[key0]
        tot0 += n * ((1 - P0) * rA - ETA * W0)
    return tot / ns.sum(), tot0 / ns.sum()

def optimize(theta, model='mixed', grid=None, hours=range(24), arr=None, verbose=False, **kw):
    arr = arrivals_by_hour() if arr is None else arr
    grid = np.round(np.arange(0, 1.0001, 0.05), 3) if grid is None else grid
    best = {}; curves = {}
    for h in hours:
        vals = np.array([hour_value2(h, f, arr[h], theta, model, **kw)[0] - hour_value2(h, 0, arr[h], theta, model, **kw)[1] for f in grid])
        best[h] = (grid[vals.argmax()], vals.max()); curves[h] = vals
        if verbose: print(f'{h:02d} best f={grid[vals.argmax()]:.2f} val={vals.max():+.4f}')
    return best, curves

def schedule_value(sched, theta, model='mixed', arr=None, **kw):
    arr = arrivals_by_hour() if arr is None else arr
    tot = 0.0; tot0 = 0.0; N = 0
    for h in range(24):
        v, v0 = hour_value2(h, sched[h], arr[h], theta, model, **kw)
        n = arr[h].sum(); tot += n * v; tot0 += n * v0; N += n
    return (tot - tot0) / N

THETA = 0.01686
RB_PEAK, RB_OFF = 0.67624, 0.66015
RA = 0.64209
def dB_h(h): return (RB_PEAK if 9 <= h <= 17 else RB_OFF) - RA
MB_PEAK, MB_OFF = 6.042, 4.674

def full_days():
    import csv as _csv, glob
    rows = load_rows(); dow = {}
    for r in _csv.DictReader(open('data/history.csv')): dow[int(r['day'])] = r['dow']
    for fn in glob.glob('data/abtest_*.csv'):
        for r in _csv.DictReader(open(fn)): dow[int(r['day'])] = r['dow']
    byday = collections.defaultdict(dict)
    for r in rows: byday[r['day']][r['hour']] = r['n']
    full = [d for d in sorted(byday) if len(byday[d]) == 24]
    N = np.array([[byday[d][h] for h in range(24)] for d in full], float)
    return full, [dow[d] for d in full], N

def parametric_arrivals(N, dows, mode='dow', ndays=2000, seed=1, sigma=None):
    """simulate day arrival counts: Poisson(P_h * g_dow * lognormal)"""
    logN = np.log(N); a = logN.mean(0); b = (logN - a).mean(1)
    D = np.exp(b)
    dows = np.array(dows)
    if mode == 'dow':
        g = {k: D[dows == k].mean() for k in set(dows)}
    else:
        wk = np.isin(dows, ['Sat', 'Sun'])
        g = {k: (D[wk].mean() if k in ('Sat', 'Sun') else D[~wk].mean()) for k in set(dows)}
    resid = np.array([D[i] / g[dows[i]] for i in range(len(D))])
    s = np.log(resid).std(ddof=len(g)) if sigma is None else sigma
    P = np.exp(a) * np.exp(-np.log(resid).mean())  # profile such that day factor ~ g
    rng = np.random.default_rng(seed)
    order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    out = np.zeros((ndays, 24))
    for i in range(ndays):
        k = order[i % 7]
        fac = g[k] * np.exp(s * rng.standard_normal() - s * s / 2)
        out[i] = rng.poisson(P * fac)
    return out, g, s, P

def arr_from_N(N): return {h: N[:, h] for h in range(24)}
