import numpy as np, math
def erlang_a(lam, mu, c, theta, nmax=4000):
    """M/M/c+M steady state. lam arrivals/s, mu service rate, c servers, theta abandon hazard.
    returns (P_abandon, E[W] over all arrivals, utilization)"""
    # log probabilities
    logp = np.zeros(nmax+1)
    a = lam/mu
    for n in range(1, nmax+1):
        if n <= c:
            logp[n] = logp[n-1] + math.log(a) - math.log(n)
        else:
            logp[n] = logp[n-1] + math.log(lam) - math.log(c*mu + (n-c)*theta)
        if n > c and logp[n] < logp.max() - 40: 
            logp = logp[:n+1]; break
    p = np.exp(logp - logp.max()); p /= p.sum()
    n = np.arange(len(p))
    busy = np.minimum(n, c)
    thr = mu * (p*busy).sum()
    pab = 1 - thr/lam
    EW = pab/theta
    util = (p*busy).sum()/c
    return pab, EW, util
