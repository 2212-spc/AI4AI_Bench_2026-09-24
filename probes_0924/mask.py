import numpy as np, math, sys, itertools, time
D,H,T_H=32,128,64
def teacher(seed=0):
    r=np.random.default_rng(1000+seed)
    W1=r.normal(0,1/math.sqrt(D),(D,T_H)); W2=r.normal(0,1/math.sqrt(T_H),(T_H,1))
    return lambda X: np.tanh(X@W1)@W2
f=teacher()
NOISE=0.1
def data(n,seed):
    r=np.random.default_rng(seed); X=r.normal(size=(n,D)); y=f(X)+NOISE*r.normal(size=(n,1)); return X,y
Xtr,ytr=data(20000,1); Xva,yva=data(4000,2)
def run(eta,ga_bug,sched_bug,seed,steps=600,k=4,mb=16,warm=30,minr=0.1,mom=0.9):
    r=np.random.default_rng(seed)
    W1=r.normal(0,1/math.sqrt(D),(D,H)); b1=np.zeros(H); W2=r.normal(0,1/math.sqrt(H),(H,1)); b2=np.zeros(1)
    P=[W1,b1,W2,b2]; V=[np.zeros_like(p) for p in P]
    def lr_at(t):
        if t<warm: return eta*(t+1)/warm
        if t>steps: return eta*minr
        c=0.5*(1+math.cos(math.pi*(t-warm)/(steps-warm))); return eta*(minr+(1-minr)*c)
    sched_t=0; lr=lr_at(0)
    for s in range(steps):
        G=[np.zeros_like(p) for p in P]
        for m in range(k):
            idx=r.integers(0,len(Xtr),mb); X=Xtr[idx]; y=ytr[idx]
            h=np.maximum(X@W1+b1,0); out=h@W2+b2; d=(out-y)*2/mb
            gW2=h.T@d; gb2=d.sum(0); dh=(d@W2.T)*(h>0); gW1=X.T@dh; gb1=dh.sum(0)
            sc=1.0 if ga_bug else 1.0/k
            for g,gg in zip(G,[gW1,gb1,gW2,gb2]): g+=sc*gg
            if sched_bug: sched_t+=1; lr=lr_at(sched_t)
        if not sched_bug: sched_t+=1; lr_use=lr_at(s)
        else: lr_use=lr
        for p,v,g in zip(P,V,G):
            v*=mom; v+=g; p-=lr_use*v
        if not np.isfinite(W1).all() or np.abs(W1).max()>1e6: return float('inf')
    out=np.maximum(Xva@W1+b1,0)@W2+b2
    return float(((out-yva)**2).mean())
if __name__=="__main__":
    t=time.time()
    for eta in [float(x) for x in sys.argv[1:]]:
        res={}
        for ga,sb in itertools.product([1,0],[1,0]):
            v=[run(eta,ga,sb,s) for s in range(3)]
            res[(ga,sb)]=v
        fmt=lambda v: f"{np.mean(v):.4f}±{np.std(v):.4f}"
        print(f"eta={eta}: BOTH_BUGS {fmt(res[(1,1)])} | fixGA_only {fmt(res[(0,1)])} | fixSched_only {fmt(res[(1,0)])} | FIXED {fmt(res[(0,0)])}",flush=True)
    print("bayes mse",NOISE**2, "time",time.time()-t)
