import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,time,numpy as np
sys.path.insert(0,'/app/experiments')
import original_train as tr
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz')
rng=np.random.default_rng(27);ix=rng.permutation(len(s['y']))
cfg=json.load(open('/app/repo/config.json'))
for n in [1000,2000,4000]:
 for wd in [0.15,0.3,0.7,1.5,3.,6.]:
  cfg['weight_decay']=wd
  t=time.process_time();p,norm=tr.train(s['X'][ix[:n]],s['y'][ix[:n]],n//4,cfg)
  acc=(tr.predict(p,norm,d['X'])==d['y']).mean()
  trainacc=(tr.predict(p,norm,s['X'][ix[:n]])==s['y'][ix[:n]]).mean()
  print(json.dumps(dict(n=n,wd=wd,acc=acc,trainacc=trainacc,cpu=time.process_time()-t)),flush=True)
