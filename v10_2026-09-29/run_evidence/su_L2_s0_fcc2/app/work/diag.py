import sys, json, numpy as np
sys.path.insert(0,'/app/repo'); import train as T
S=np.load('/app/data/sample.npz'); D=np.load('/app/data/dev.npz')
X,y=S['X'].astype(np.float32),S['y'].astype(np.int64); Xd,yd=D['X'].astype(np.float32),D['y']
cfg=json.load(open('/app/repo/config.json')); K=10
n=int(sys.argv[1]); grid=[0.1,0.3,1,2,3,6]
for s in range(3):
    idx=np.random.default_rng(1000+s).choice(4000,n,replace=False); Xn,yn=X[idx],y[idx]
    norm=T.standardizer(Xn); Xs=norm(Xn); Xds=norm(Xd)
    perm=np.random.default_rng(10000+s).permutation(n); nh=int(0.15*n); hold,rest=perm[:nh],perm[nh:]
    steps=int(32*n/128)
    for frac in [0.5,1.0]:
        rows=[]
        for w in grid:
            P=T.fit(Xs[rest],yn[rest],int(frac*steps),cfg,K,s*100,w)
            z=T.logits([P],Xs[hold]); acc=(z.argmax(1)==yn[hold]).mean()
            z=z-z.max(1,keepdims=True); lp=z-np.log(np.exp(z).sum(1,keepdims=True)); ll=-lp[np.arange(nh),yn[hold]].mean()
            g=(T.logits([P],Xds).argmax(1)==yd).mean()
            rows.append((w,acc,ll,g))
        print(f"n={n} seed={s} frac={frac}: "+"  ".join(f"wd{w}: h{a:.3f}/ll{l:.3f}/g{g:.3f}" for w,a,l,g in rows), flush=True)
