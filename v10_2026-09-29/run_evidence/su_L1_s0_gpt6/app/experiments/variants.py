import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,time,json,numpy as np
sys.path.insert(0,'/app/repo')
import train as m
D=np.load('/app/data/sample.npz');V=np.load('/app/data/dev.npz'); X,y=D['X'],D['y'];vx,vy=V['X'],V['y']
cfg=json.load(open('/app/repo/config.json'))
def fit(X,y,steps,cfg,seed=0):
 rng=np.random.default_rng(seed); k=10; norm=m.standardizer(X); xs=norm(X)
 P=m.init_params(xs.shape[1],cfg['hidden'],k,rng);opt=m.AdamW(P,cfg['lr'],cfg['weight_decay'])
 Ps={n:np.zeros_like(v) for n,v in P.items()};ns=0
 logits=[]
 for t in range(steps):
  idx=rng.integers(0,len(X),cfg['batch_size']);xb,yb=xs[idx],y[idx]
  if cfg['aug_sigma']>0:xb=xb+cfg['aug_sigma']*rng.standard_normal(xb.shape).astype(np.float32)
  z,cache=m.forward(P,xb)
  p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);py=p[np.arange(len(yb)),yb].copy()
  p[np.arange(len(yb)),yb]-=1
  if cfg.get('noise',0):
   noise=cfg['noise']*min(1,t/(.2*steps))
   p*=((1-noise)*py/((1-noise)*py+noise/k))[:,None]
  if cfg.get('q',0): p*=py[:,None]**cfg['q']
  G=m.backward(P,cache,p/len(yb)); opt.step(P,G,m.lr_multiplier(t,steps,cfg['warmup_frac']))
  if (t+1) % max(1,steps//10)==0 and t>=steps*.5:
   ns+=1
   for name in P:Ps[name]+=P[name]
   zz,_=m.forward(P,norm(vx));logits.append(zz)
 for name in P:Ps[name]/=ns
 return np.mean(m.predict(P,norm,vx)==vy),np.mean(m.predict(Ps,norm,vx)==vy),np.mean(np.mean(logits,axis=0).argmax(1)==vy),np.mean(m.predict(P,norm,X)==y)
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='loss':
  variants=[dict(noise=noise,weight_decay=wd) for noise in [0.,.15,.3,.45] for wd in [.5,1.,2.]]
 elif mode=='arch':
  variants=[dict(hidden=h,weight_decay=wd) for h in [64,128,256,512] for wd in [.5,1.,2.]]
 elif mode=='aug':
  variants=[dict(aug_sigma=aug,weight_decay=wd) for aug in [.1,.2,.4] for wd in [.5,1.,2.]]
 for v in variants:
  c=dict(cfg,**v); t=time.time(); result=fit(X,y,1000,c,0)
  print(v, 'final/weights/logits/train',np.round(result,4),'seconds',round(time.time()-t,2),flush=True)
