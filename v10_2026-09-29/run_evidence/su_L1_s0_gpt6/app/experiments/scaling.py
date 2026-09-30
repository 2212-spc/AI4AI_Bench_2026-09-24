import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,time,json,numpy as np
sys.path.insert(0,'/app/repo')
import train as m
D=np.load('/app/data/sample.npz');V=np.load('/app/data/dev.npz'); X,y=D['X'],D['y'];vx,vy=V['X'],V['y']
cfg=json.load(open('/app/repo/config.json'))
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
inds=np.random.default_rng(456+seed).permutation(len(X))
res=[]
for n in [500,1000,2000,4000]:
 for wd in [.25,.5,1.,2.,4.,8.]:
  c=dict(cfg,weight_decay=wd);t=time.time(); P,norm=m.train(X[inds[:n]],y[inds[:n]],n//4,c,seed=seed)
  row=dict(n=n,wd=wd,seed=seed,dev=float(np.mean(m.predict(P,norm,vx)==vy)),train=float(np.mean(m.predict(P,norm,X[inds[:n]])==y[inds[:n]])))
  res.append(row);print(row,flush=True)
json.dump(res,open('/app/experiments/scaling%d.json'%seed,'w'))
