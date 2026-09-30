import numpy as np, csv, math
from math import lgamma, log, exp

def erlang_a(lam, mu, c, theta):
    """M/M/c+M steady state. Returns (P_abandon, E[W] over all arrivals, P(wait>0)).
    lam arrival rate, mu service rate per server, theta abandonment hazard."""
    if lam <= 0: return 0.0, 0.0, 0.0
    a = lam/mu
    # log pi_n unnormalized for n<=c: a^n/n!
    logp = [n*log(a) - lgamma(n+1) for n in range(c+1)]
    # for n>c: pi_n = pi_c * prod_{k=1}^{n-c} lam/(c mu + k theta)
    tail = []
    lp = logp[c]
    n = c
    while True:
        n += 1
        lp += log(lam) - log(c*mu + (n-c)*theta)
        tail.append(lp)
        if lp < logp[c] - 40 and n > c + 5: break
        if n > c + 200000: break
    allp = np.array(logp + tail)
    m = allp.max()
    p = np.exp(allp - m); p /= p.sum()
    ns = np.arange(len(p))
    q = np.where(ns > c, ns - c, 0)
    EQ = (p*q).sum()          # expected queue length
    # abandonment rate = theta*E[Q]; P_ab = theta EQ / lam ; E[W] = EQ/lam (Little)
    EW = EQ/lam
    Pab = theta*EW
    Pwait = p[c:].sum()  # prob arrival finds all busy (PASTA)
    return Pab, EW, Pwait

def replicas(lam, mean_gen, cap=24):
    load = lam*mean_gen
    return max(4, min(cap, math.ceil(load/0.75 - 1e-12)))

if __name__ == "__main__":
    rows=list(csv.DictReader(open('/app/data/history.csv')))
    theta=0.0419; mA=3.57
    bad=0
    res=[]
    for r in rows:
        lam=float(r['requests'])/3600
        c=replicas(lam,mA)
        if c!=int(r['replicas']): bad+=1
        Pab,EW,_=erlang_a(lam,1/mA,int(r['replicas']),theta)
        res.append((int(r['hour']),float(r['abandon_rate']),Pab,float(r['mean_queue_wait_s']),EW,float(r['utilization']), lam*mA*(1-Pab)/int(r['replicas'])))
    print('replica mismatches',bad,len(rows))
    res=np.array(res)
    for h in range(24):
        s=res[res[:,0]==h]
        print(h, 'ab obs %.4f pred %.4f | w obs %.3f pred %.3f | util obs %.3f pred %.3f'%(s[:,1].mean(),s[:,2].mean(),s[:,3].mean(),s[:,4].mean(),s[:,5].mean(),s[:,6].mean()))
    print('ratio ab', (res[:,1]/res[:,2]).mean(), 'ratio w', (res[:,3]/res[:,4]).mean())
