import numpy as np
from math import ceil, lgamma, log, exp

def erlang_a(lam, mu, c, theta, nmax=4000):
    """M/M/c+M steady state. Returns (P_abandon, E[W] over all arrivals, utilization)."""
    # log probabilities unnormalized
    # p_n = (lam/mu)^n / n!  for n<=c ; p_{c+k} = p_c * prod_{j=1..k} lam/(c mu + j theta)
    a = lam/mu
    logp = np.empty(nmax+1)
    for n in range(c+1):
        logp[n] = n*log(a) - lgamma(n+1)
    cur = logp[c]
    for k in range(1, nmax-c+1):
        cur += log(lam) - log(c*mu + k*theta)
        logp[c+k] = cur
        if cur < logp.max()-60 and k>50: 
            logp = logp[:c+k+1]; break
    m = logp.max()
    p = np.exp(logp-m); p /= p.sum()
    n = np.arange(len(p))
    q = np.maximum(n-c, 0)
    EQ = (p*q).sum()
    # abandonment rate = theta * E[Q]
    pab = theta*EQ/lam
    EW = EQ/lam  # Little's law: E[W] over all arrivals = E[Q]/lam
    busy = (p*np.minimum(n,c)).sum()
    return pab, EW, busy/c

def replicas(lam, sbar, cap=32):
    return int(min(cap, max(4, ceil(lam*sbar/0.75 - 1e-12))))
