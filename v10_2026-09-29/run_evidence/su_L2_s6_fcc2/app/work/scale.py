import numpy as np, sys, json, itertools
sys.path.insert(0,'/app/repo'); import train as T
s=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz')
X,y=s['X'],s['y']; Xd,yd=d['X'],d['y']
base=json.load(open('/app/repo/config.json'))
def run(n, cfg, seeds=(0,1,2), sub_seed=0):
    accs=[]
    for sd in seeds:
        r=np.random.default_rng(100+sub_seed+sd); idx=r.choice(len(X),n,replace=False) if n<len(X) else np.arange(len(X))
        steps=int(32*n/cfg['batch_size'])
        P,norm=T.train(X[idx],y[idx],steps,cfg,seed=sd)
        accs.append((T.predict(P,norm,Xd)==yd).mean())
    return np.mean(accs)
for n in [500,1000,2000,4000]:
    for wd in [0.3,1.0,2.0,3.0,5.0]:
        cfg=dict(base,weight_decay=wd)
        print(n,wd,round(run(n,cfg),4),flush=True)
