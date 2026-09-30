import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time,numpy as np
sys.path.insert(0,'/app/experiments')
import original_train as tr
noise=.25
def grad(z,y,k,smoothing):
 z=z-z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
 py=p[np.arange(len(y)),y].copy();obs=(1-noise)*py+noise/k
 w=(1-noise)*py/obs;p[np.arange(len(y)),y]-=1
 return -np.log(obs).mean(),p*(w/len(y))[:,None]
tr.softmax_xent_grad=grad
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');cfg=json.load(open('/app/repo/config.json'))
for h,wd in [(128,1.),(256,1.5),(256,3.)]:
 cfg.update(hidden=h,weight_decay=wd)
 for steps in [1000,4000]:
  p,norm=tr.train(s['X'],s['y'],steps,cfg)
  print(h,wd,steps,(tr.predict(p,norm,d['X'])==d['y']).mean(),flush=True)
