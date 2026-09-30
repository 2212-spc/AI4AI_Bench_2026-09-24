import numpy as np, math, csv, collections
def erlang_a(lam, mu, c, theta, nmax=4000):
    # returns (abandon_prob, mean_wait_all_arrivals, utilization)
    # log-space unnormalized probs
    logp=[0.0]
    a=lam/mu
    for n in range(1,c+1):
        logp.append(logp[-1]+math.log(a/n))
    n=c
    while True:
        n+=1
        k=n-c
        logp.append(logp[-1]+math.log(lam/(c*mu+k*theta)))
        if logp[-1] < logp[c]-40 and k>10: break
        if n>nmax: break
    lp=np.array(logp); m=lp.max(); p=np.exp(lp-m); p/=p.sum()
    ns=np.arange(len(p))
    EQ=np.sum(np.maximum(ns-c,0)*p)
    busy=np.sum(np.minimum(ns,c)*p)
    ab=theta*EQ/lam
    W=EQ/lam
    return ab,W,busy/c
def replicas(offered, cap=16):
    return int(min(cap,max(4,math.ceil(offered/0.75))))
if __name__=='__main__':
    rows=list(csv.DictReader(open('/app/data/history.csv')))
    theta=0.0218
    for r in rows[:24]:
        lam=float(r['requests'])/3600; mu=1/3.076
        c=int(r['replicas'])
        ab,W,u=erlang_a(lam,mu,c,theta)
        print(r['hour'],c,replicas(lam*3.076),'ab',r['abandon_rate'],round(ab,5),'W',r['mean_queue_wait_s'],round(W,3),'u',r['utilization'],round(u,3))
