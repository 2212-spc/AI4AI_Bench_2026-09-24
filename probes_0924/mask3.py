import numpy as np, math, itertools, time
from mask import Xtr,ytr,Xva,yva,D,H
def run(eta,ga_bug,damp_bug,seed,steps=600,k=4,mb=16,warm=30,minr=0.1,mom=0.9,clip=None):
    r=np.random.default_rng(seed)
    W1=r.normal(0,1/math.sqrt(D),(D,H)); b1=np.zeros(H); W2=r.normal(0,1/math.sqrt(H),(H,1)); b2=np.zeros(1)
    P=[W1,b1,W2,b2]; V=[np.zeros_like(p) for p in P]
    def lr_at(t):
        if t<warm: return eta*(t+1)/warm
        c=0.5*(1+math.cos(math.pi*min(1,(t-warm)/(steps-warm)))); return eta*(minr+(1-minr)*c)
    for s in range(steps):
        G=[np.zeros_like(p) for p in P]
        for m in range(k):
            idx=r.integers(0,len(Xtr),mb); X=Xtr[idx]; y=ytr[idx]
            h=np.maximum(X@W1+b1,0); out=h@W2+b2; d=(out-y)*2/mb
            gW2=h.T@d; gb2=d.sum(0); dh=(d@W2.T)*(h>0); gW1=X.T@dh; gb1=dh.sum(0)
            sc=1.0 if ga_bug else 1.0/k
            for g,gg in zip(G,[gW1,gb1,gW2,gb2]): g+=sc*gg
        if clip:
            n=math.sqrt(sum(float((g*g).sum()) for g in G))
            if n>clip: G=[g*clip/n for g in G]
        lr=lr_at(s)
        for p,v,g in zip(P,V,G):
            v*=mom; v+=(1-mom)*g if damp_bug else g; p-=lr*v
        if not np.isfinite(W1).all() or np.abs(W1).max()>1e6: return float('inf')
    out=np.maximum(Xva@W1+b1,0)@W2+b2
    return float(((out-yva)**2).mean())
t=time.time()
for clip in [None,5.0,2.0]:
  for eta in [0.04,0.06,0.08,0.12]:
    R={}
    for ga,db in itertools.product([1,0],[1,0]):
        R[(ga,db)]=np.array([run(eta,ga,db,s,clip=clip) for s in range(4)])
    m={k:v.mean() for k,v in R.items()}; s={k:v.std() for k,v in R.items()}
    fin=all(np.isfinite(R[(1,1)]))
    mask = fin and m[(0,0)]<m[(1,1)]<min(m[(0,1)],m[(1,0)])
    print(f"clip={clip} eta={eta}: CURRENT(both) {m[(1,1)]:.4f}±{s[(1,1)]:.4f} | fixGA_only {m[(0,1)]:.4f} | fixDamp_only {m[(1,0)]:.4f} | FIXED {m[(0,0)]:.4f}±{s[(0,0)]:.4f}  MASK={mask}",flush=True)
print(time.time()-t)
