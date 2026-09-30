import numpy as np, sys, json, time
sys.path.insert(0,'/app/repo'); import train as T
s=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz')
X,y=s['X'],s['y']; Xd,yd=d['X'],d['y']
BASE=json.load(open('/app/repo/config.json'))
hid=int(sys.argv[1]); lr=float(sys.argv[2])
for n in [1000,2000,3000]:
    for wd in [0.1,0.3,0.6,1.0,1.5,2.0,3.0,4.0]:
        cfg=dict(BASE,weight_decay=wd,hidden=hid,lr=lr); g=[]; nz=[]
        for sd in range(4):
            r=np.random.default_rng(100+sd); perm=r.permutation(len(X)); idx=perm[:n]; ho=perm[3000:]
            P,norm=T.train(X[idx],y[idx],int(32*n/128),cfg,seed=sd)
            g.append((T.predict(P,norm,Xd)==yd).mean()); nz.append((T.predict(P,norm,X[ho])==y[ho]).mean())
        print(hid,lr,n,wd,'gold %.4f noisy %.4f'%(np.mean(g),np.mean(nz)),flush=True)
