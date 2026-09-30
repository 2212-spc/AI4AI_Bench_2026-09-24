import sys, json, time
sys.path.insert(0,'/app/repo')
import train
import numpy as np
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');cfg=json.load(open('/app/repo/config.json'))
rng=np.random.default_rng(93); order=rng.permutation(len(s['y']))
for n in [1000,2000,4000]:
 for wd in [0.15,0.3,0.6,1.,2.,3.,5.,8.]:
  acc=[]; t=time.process_time()
  for seed in [0,1]:
   c=dict(cfg,weight_decay=wd)
   p,norm=train.train(s['X'][order[:n]],s['y'][order[:n]],n//4,c,seed)
   acc.append(float(np.mean(train.predict(p,norm,d['X'])==d['y'])))
  print(json.dumps(dict(n=n,wd=wd,acc=acc,cpu=time.process_time()-t)),flush=True)
