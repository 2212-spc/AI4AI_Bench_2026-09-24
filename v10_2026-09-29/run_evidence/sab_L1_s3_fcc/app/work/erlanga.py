import numpy as np
def erlang_a(lam, mu, c, theta, nmax=4000):
    """M/M/c+M steady state. lam arrivals/s, mu service rate/s, c servers, theta abandon hazard.
    returns dict: p_ab, EW (mean wait over all arrivals), util, EQ"""
    # unnormalized log-probs
    n = np.arange(nmax+1)
    down = np.where(n<=c, n*mu, c*mu + (n-c)*theta)
    logp = np.zeros(nmax+1)
    logp[1:] = np.cumsum(np.log(lam) - np.log(down[1:]))
    logp -= logp.max()
    p = np.exp(logp); p /= p.sum()
    q = np.maximum(n-c,0)
    EQ = (p*q).sum()
    EW = EQ/lam
    p_ab = theta*EQ/lam
    busy = (p*np.minimum(n,c)).sum()
    return dict(p_ab=p_ab, EW=EW, util=busy/c, EQ=EQ, p_wait=p[c:].sum())
