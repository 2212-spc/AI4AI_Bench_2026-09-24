import sys,json,itertools
sys.path.insert(0,'/app/repo');import train
import numpy as np
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');base=json.load(open('/app/repo/config.json'));rng=np.random.default_rng(93)
for n in [2000,4000]:
 idx=rng.permutation(4000)[:n]
 for lr,wd,sm in itertools.product([.001,.0015,.002,.003],[1.,2.,3.,4.],[0,.08]):
  c=dict(base,lr=lr,weight_decay=wd,label_smoothing=sm)
  aa=[]
  for seed in [0,1]:
   p,nn=train.train(s['X'][idx],s['y'][idx],n//4,c,seed)
   aa.append(float(np.mean(train.predict(p,nn,d['X'])==d['y'])))
  print(json.dumps(dict(n=n,lr=lr,wd=wd,sm=sm,a=aa,m=np.mean(aa))),flush=True)
