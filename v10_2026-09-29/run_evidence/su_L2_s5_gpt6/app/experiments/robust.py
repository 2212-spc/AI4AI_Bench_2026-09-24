import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import numpy as np,sys,json,time
sys.path.insert(0,'/app/experiments'); import baseline as base

def normer(X,white=0):
 mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd
 if white:
  v,U=np.linalg.eigh(Z.T@Z/len(X)); A=((U*np.maximum(v,0.02)**(-0.5*white))@U.T).astype(np.float32)
  return lambda X: np.asarray(((X-mu)/sd)@A,dtype=np.float32)
 return lambda X: np.asarray((X-mu)/sd,dtype=np.float32)

def probabilities(P,norm,X):
 z=base.forward(P,norm(X))[0]; z-=z.max(1,keepdims=True); p=np.exp(z); return p/p.sum(1,keepdims=True)

def train(X,y,steps,cfg,seed=0):
 rng=np.random.default_rng(seed); k=10; norm=normer(X,cfg.get('white',0)); Z=norm(X); P=base.init_params(X.shape[1],cfg['hidden'],k,rng); opt=base.AdamW(P,cfg['lr'],cfg['weight_decay']); bs=cfg['batch_size']; rows=np.arange(bs)
 eta=cfg.get('noise',0); avg=None; ac=0
 for t in range(steps):
  ids=rng.integers(len(Z),size=bs); xb=Z[ids]; yb=y[ids]
  if cfg.get('aug_sigma',0): xb=xb+cfg['aug_sigma']*rng.standard_normal(xb.shape).astype(np.float32)
  z,c=base.forward(P,xb); z-=z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True)
  py=p[rows,yb].copy(); dz=p; dz[rows,yb]-=1
  if eta: dz*=((1-eta)*py/((1-eta)*py+eta/k))[:,None]
  dz/=bs; G=base.backward(P,c,dz); opt.step(P,G,base.lr_multiplier(t,steps,cfg['warmup_frac']))
  if cfg.get('avg',0) and t>=int(steps*(1-cfg['avg'])):
   ac+=1
   if avg is None: avg={n:v.copy() for n,v in P.items()}
   else:
    for n in P: avg[n]+=(P[n]-avg[n])/ac
 return avg if avg is not None else P,norm

if __name__=='__main__':
 s=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz'); X,y=s['X'],s['y']; D,Y=d['X'],d['y']; cfg=json.load(open('/app/repo/config.json')); results=[]
 for white in [0,0.5,1.0]:
  for eta in [0,0.3]:
   for wd in ([1,2,3,5] if eta==0 else [0.3,0.7,1,2,3]):
    c=dict(cfg,white=white,noise=eta,weight_decay=wd); tt=time.process_time(); acc=[]; pr=[]
    for seed in [0,1]:
     P,norm=train(X,y,1000,c,seed); p=probabilities(P,norm,D); pr.append(p); acc.append(float((p.argmax(1)==Y).mean()))
    r=dict(white=white,noise=eta,wd=wd,dev=acc,ens=float((np.mean(pr,0).argmax(1)==Y).mean()),cpu=round(time.process_time()-tt,2)); results.append(r); print(r,flush=True)
 json.dump(results,open('/app/experiments/robust.json','w'),indent=2)
