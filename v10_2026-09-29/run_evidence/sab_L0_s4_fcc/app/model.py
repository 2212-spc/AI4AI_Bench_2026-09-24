import numpy as np, csv, math

def erlang_a(lam, mu, c, theta, nmax=None):
    """M/M/c+M stationary: returns (P_abandon, E[W_all], E[Q], P_wait). lam,mu,theta per second."""
    if nmax is None:
        nmax = int(c + 20*max(1.0, lam/theta) + 200)
    # log stationary probs via recursion p_{n+1}/p_n = lam / d(n+1)
    logp = np.zeros(nmax+1)
    for n in range(nmax):
        d = min(n+1, c)*mu + max(n+1-c, 0)*theta
        logp[n+1] = logp[n] + math.log(lam) - math.log(d)
    logp -= logp.max()
    p = np.exp(logp); p /= p.sum()
    n = np.arange(nmax+1)
    EQ = np.sum(np.maximum(n-c, 0)*p)
    Pab = theta*EQ/lam
    EW = EQ/lam
    Pwait = p[n >= c].sum()
    util = np.sum(np.minimum(n, c)*p)/c
    return Pab, EW, EQ, Pwait, util

def replicas(lam, mean_gen, cap=32):
    return int(min(cap, max(4, math.ceil(lam*mean_gen/0.75 - 1e-9))))

if __name__ == "__main__":
    rows = list(csv.DictReader(open('data/history.csv')))
    ab = np.array([float(r['abandon_rate']) for r in rows])
    w = np.array([float(r['mean_queue_wait_s']) for r in rows])
    ratio = ab/w
    print("theta est: mean ratio %.5f  median %.5f  se %.5f  -> patience %.1fs" % (ratio.mean(), np.median(ratio), ratio.std()/np.sqrt(len(ratio)), 1/ratio.mean()))
    # weighted
    print("theta sum-ratio: %.5f" % (ab.sum()/w.sum()))
    gen = np.array([float(r['mean_gen_time_s']) for r in rows])
    req = np.array([float(r['requests']) for r in rows])
    print("gen time A: %.4f (weighted %.4f)" % (gen.mean(), (gen*req).sum()/req.sum()))
    rat = np.array([float(r['mean_rating']) for r in rows])
    print("rating A: %.5f weighted %.5f  sd across hours %.4f" % (rat.mean(), (rat*req).sum()/req.sum(), rat.std()))
    # check replicas rule
    bad = 0
    for r in rows:
        lam = float(r['requests'])/3600
        c = replicas(lam, 1.95)
        if c != int(r['replicas']): bad += 1
    print("replica mismatches with mean_gen=1.95:", bad, "of", len(rows))
    # check model predictions
    theta = ab.sum()/w.sum()
    mu = 1/1.95
    pred_ab=[]; pred_w=[]
    for r in rows:
        lam = float(r['requests'])/3600; c=int(r['replicas'])
        Pab, EW, EQ, Pw, u = erlang_a(lam, mu, c, theta)
        pred_ab.append(Pab); pred_w.append(EW)
    pred_ab=np.array(pred_ab); pred_w=np.array(pred_w)
    print("abandon obs/pred mean ratio: %.4f ; wait obs/pred: %.4f" % ((ab/pred_ab).mean(), (w/pred_w).mean()))
    print("corr:", np.corrcoef(ab, pred_ab)[0,1])
    # score check
    sc = np.array([float(r['mean_score']) for r in rows])
    pred_sc = rat*(1-ab) - 0.0069*w
    print("score resid mean %.5f sd %.5f" % ((sc-pred_sc).mean(), (sc-pred_sc).std()))

def mix_model(lam, f, gA, gB, theta, cap=32):
    """approximate mixed-traffic hour: use Erlang-A with mean gen of the mix (exponential approx)."""
    g = (1-f)*gA + f*gB
    c = replicas(lam, g, cap)
    Pab, EW, EQ, Pw, u = erlang_a(lam, 1/g, c, theta)
    return c, Pab, EW, u
