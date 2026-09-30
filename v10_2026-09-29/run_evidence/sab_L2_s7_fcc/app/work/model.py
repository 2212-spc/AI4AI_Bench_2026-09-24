import numpy as np, csv, math
THETA=0.017124
def erlangA(lam, m, c, theta=THETA, nmax=3000):
    """lam arrivals/s, m mean service s, c servers. returns (P_ab, E[W])"""
    mu=1.0/m
    n=np.arange(1,nmax+1)
    rates=np.where(n<=c, n*mu, c*mu+(n-c)*theta)
    logp=np.concatenate([[0.0],np.cumsum(np.log(lam/rates))])
    p=np.exp(logp-logp.max()); p/=p.sum()
    nn=np.arange(nmax+1)
    EQ=np.sum(np.maximum(nn-c,0)*p)
    pab=theta*EQ/lam
    return pab, EQ/lam
def replicas(lam, m, cap=16):
    return int(min(cap, max(4, math.ceil(lam*m/0.75 - 1e-9))))
