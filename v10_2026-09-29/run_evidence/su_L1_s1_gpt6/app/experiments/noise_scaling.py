import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time,numpy as np
sys.path.insert(0,'/app/experiments')
import original_train as tr
noise=.25
def noise_grad(z,y,k,smoothing):
 z=z-z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
 py=p[np.arange(len(y)),y].copy();obs=(1-noise)*py+noise/k
 w=(1-noise)*py/obs;p[np.arange(len(y)),y]-=1
 return -np.log(obs).mean(),p*(w/len(y))[:,None]
tr.softmax_xent_grad=noise_grad
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');cfg=json.load(open('/app/repo/config.json'))
ix=np.random.default_rng(27).permutation(4000)
for n in [1000,2000,4000]:
 for h in [128,256]:
  for basewd in [.5,1.,2.]:
   wd=basewd*np.sqrt(4000/n)*(h/128)
   cfg.update(hidden=h,weight_decay=wd)
   p,norm=tr.train(s['X'][ix[:n]],s['y'][ix[:n]],n//4,cfg,seed=0)
   print(json.dumps(dict(n=n,h=h,basewd=basewd,wd=wd,acc=(tr.predict(p,norm,d['X'])==d['y']).mean())),flush=True)
