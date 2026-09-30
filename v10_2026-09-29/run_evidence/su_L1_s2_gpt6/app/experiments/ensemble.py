import sys,json,time
sys.path.insert(0,'/app/repo')
import train
import numpy as np
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');cfg=json.load(open('/app/repo/config.json'))
for n in [1000,2000,4000]:
 rng=np.random.default_rng(93);idx=rng.permutation(4000)[:n]
 for wds in [[1.5,2.5,4.],[1.,2.,3.],[2.,3.,4.]]:
  zs=[]; acc=[]
  for seed,wd in enumerate(wds):
   c=dict(cfg,weight_decay=wd*(4000/n)**0.6)
   p,norm=train.train(s['X'][idx],s['y'][idx],n//4,c,seed)
   z,_=train.forward(p,norm(d['X']));zs.append(z);acc.append(float(np.mean(z.argmax(1)==d['y'])))
  print(json.dumps(dict(n=n,base_wds=wds,acc=acc,ensemble=float(np.mean(np.mean(zs,axis=0).argmax(1)==d['y'])))),flush=True)
