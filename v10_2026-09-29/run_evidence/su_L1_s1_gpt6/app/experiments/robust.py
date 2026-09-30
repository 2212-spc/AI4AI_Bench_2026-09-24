import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time,numpy as np
sys.path.insert(0,'/app/experiments')
import original_train as tr
original_grad=tr.softmax_xent_grad
noise=0.
def noise_grad(z,y,k,smoothing):
 if noise==0:return original_grad(z,y,k,smoothing)
 z=z-z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
 py=p[np.arange(len(y)),y].copy();obs=(1-noise)*py+noise/k
 w=(1-noise)*py/obs
 p[np.arange(len(y)),y]-=1
 return -np.log(obs).mean(),p*(w/len(y))[:,None]
tr.softmax_xent_grad=noise_grad
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');cfg=json.load(open('/app/repo/config.json'))
for h,wd in [(128,.5),(128,1.),(256,1.5),(256,3.)]:
 for ns,aug in [(0.,0.),(.2,0.),(.3,0.),(0.,.2),(.2,.2),(0.,.4)]:
  cfg.update(hidden=h,weight_decay=wd,aug_sigma=aug);noise=ns
  ps=[];aa=[]
  for seed in [0,1]:
   p,norm=tr.train(s['X'],s['y'],1000,cfg,seed=seed)
   z,_=tr.forward(p,norm(d['X']));z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True);ps.append(p)
   aa.append((p.argmax(1)==d['y']).mean())
  print(json.dumps(dict(h=h,wd=wd,noise=ns,aug=aug,acc=aa,ensemble=(np.mean(ps,axis=0).argmax(1)==d['y']).mean())),flush=True)
