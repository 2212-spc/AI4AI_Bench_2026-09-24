import sys, json, time, numpy as np
from proxy import *
sys.path.insert(0,'/app/work'); import train2 as T2
BASE=json.load(open('/app/repo/config.json'))
PROX={'A':dict(kind='nl',K=4,spread=0.5,shrink=0.5),'lin':dict(kind='lin'),'B':dict(kind='nl',K=3,spread=0.35,shrink=0.8)}
name,n,steps,cfgs=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
noise=float(sys.argv[5]) if len(sys.argv)>5 else 0.35
nk=sys.argv[6] if len(sys.argv)>6 else 'uniform'
g=make_gen(noise=noise,noise_kind=nk,**PROX[name])
Xte,_,yte=g(20000,np.random.default_rng(999))
Xn,yc,yg=g(n,np.random.default_rng(7))
cfg=dict(BASE); cfg.update(json.loads(cfgs))
t0=time.process_time(); m,norm=T2.train(Xn,yc,steps,cfg,seed=0); dt=time.process_time()-t0
p=T2.predict(m,norm,Xte)
print(f'{name} n={n} steps={steps} noise={noise}/{nk} {cfgs}: acc={np.mean(p==yte):.4f} cpu={dt:.0f}s',flush=True)
