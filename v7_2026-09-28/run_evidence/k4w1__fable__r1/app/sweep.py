import sim, math, random, itertools
qs = [q for q in sim.load() if q['topic'] in ('algebra','geometry')]
def lnorm(x,mu,sd): return -0.5*((x-mu)/sd)**2-math.log(sd)
def llr(s):
    lc=lnorm(s,1.0,0.40); lw=math.log(0.5*math.exp(lnorm(s,1.7,0.36))+0.5*math.exp(lnorm(s,-0.45,0.72)))
    return lc-lw
def score(samples, wv, wn, wl):
    g={}
    for v,a,s in samples: g.setdefault(a,[]).append((v,s))
    out={}
    for a,l in g.items():
        n=len(l); nv=len(set(v for v,_ in l)); mean=sum(s for _,s in l)/n
        out[a]=(wl*sum(llr(s) for _,s in l)+wv*nv+wn*n, n, nv, mean)
    return out
def run_q(q, rng, cap, wv, wn, wl, stop):
    pools={v:list(q['pools'][v]) for v in range(5)}
    for p in pools.values(): rng.shuffle(p)
    samples=[]; v=0; used=0
    while len(samples)<cap:
        if len(samples)>=5:
            sc=score(samples,wv,wn,wl); b=max(sc,key=lambda a:sc[a][0]); _,n,nv,mean=sc[b]
            if stop(n,nv,mean,len(samples)): break
        p=pools[v%5]
        if not p: break
        a,s=p.pop(); samples.append((v%5,a,s)); v+=1
    sc=score(samples,wv,wn,wl); b=max(sc,key=lambda a:sc[a][0])
    return b==q['correct'], len(samples)
stops={
 'none': lambda n,nv,m,k: False,
 's1': lambda n,nv,m,k: (nv>=3 and m<1.35) or (nv>=2 and m<1.25 and k>=7),
 's2': lambda n,nv,m,k: (nv>=3 and m<1.4) or (nv>=2 and m<1.3 and k>=7),
 's3': lambda n,nv,m,k: (nv>=4 and m<1.4) or (nv>=3 and m<1.3 and k>=7),
}
for cap in (8,10,12,15):
  for wv,wn,wl in [(1.5,0.5,1),(2,0.5,1),(1.5,0,1),(2,0,1),(1.5,0.5,0.5),(1,0.5,1),(3,0,1),(1.5,0.5,2)]:
    for sn,st in stops.items():
        acc=0;use=0;N=0
        for seed in range(10):
            rng=random.Random(seed)
            for q in qs:
                ok,u=run_q(q,rng,cap,wv,wn,wl,st); acc+=ok; use+=u; N+=1
        print(f'cap={cap} wv={wv} wn={wn} wl={wl} stop={sn}: acc={acc/N:.3f} use={use/N:.2f}')
