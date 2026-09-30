import numpy as np
def erlang_a(lam, mu, c, theta, nmax=4000):
    """Return (P_abandon, E[Wq]) for M/M/c+M with arrival lam, service rate mu, c servers, patience hazard theta."""
    # birth-death: pi_{n+1} = pi_n * lam / d(n+1), d(n) = min(n,c)*mu + max(n-c,0)*theta
    logp = [0.0]
    n = 0
    while True:
        n += 1
        d = min(n, c)*mu + max(n-c, 0)*theta
        logp.append(logp[-1] + np.log(lam/d))
        if n > c and logp[-1] < logp[0] - 60 and lam/d < 1: break
        if n > nmax: break
    logp = np.array(logp); p = np.exp(logp - logp.max()); p /= p.sum()
    ns = np.arange(len(p))
    EQ = (p * np.maximum(ns - c, 0)).sum()
    EW = EQ/lam
    Pab = theta*EQ/lam
    return Pab, EW
