import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
def run(warm,reg,method):
 rng=np.random.default_rng(0);norm=b.standardizer(X);Z=norm(X);P=b.init_params(32,256,10,rng);o=b.AdamW(P,.003,3)
 for t in range(1000):
  ix=rng.integers(len(Z),size=128);z,ca=b.forward(P,Z[ix]);zz=z-z.max(1,keepdims=True);p=np.exp(zz);p/=p.sum(1,keepdims=True)
  if t<warm:
   dz=p;dz[np.arange(128),y[ix]]-=1
  else:
   if t==warm:
    zz=b.forward(P,Z)[0];zz-=zz.max(1,keepdims=True); pp=np.exp(zz);pp/=pp.sum(1,keepdims=True)
    if method=='weighted': T=(pp.T@np.eye(10)[y])/(pp.sum(0)[:,None]+1e-8)
    if method=='conf':
     T=np.zeros((10,10));
     for k in range(10):
      ix2=np.where(pp.argmax(1)==k)[0];T[k]=np.bincount(y[ix2],minlength=10)/(len(ix2)+1e-8)
    T=(T+reg*np.eye(10));T/=T.sum(1)[:,None];print(method,warm,T)
   r=p@T; fac=T[:,y[ix]].T/(r[np.arange(128),y[ix],None]+1e-8);dz=p*fac;dz-=p
  dz/=128;G=b.backward(P,ca,dz);o.step(P,G,b.lr_multiplier(t,1000,.02))
 return b.predict(P,norm,D)
for m in ['weighted','conf']:
 for warm in [200,400,600,800]:
  p=run(warm,.05,m);print(m,warm,(p==Y).mean(),flush=True)
