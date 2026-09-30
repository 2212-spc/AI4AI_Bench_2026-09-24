import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json,time
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');rng=np.random.default_rng(123);X=np.tile(s['X'],(20,1));y=np.tile(s['y'],20);q=rng.permutation(len(y));X=X[q];y=y[q];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
def tr(w,g):
 rng=np.random.default_rng(0);norm=b.standardizer(X);Z=norm(X);P=b.init_params(32,256,10,rng);o=b.AdamW(P,.003,w)
 for t in range(20000):
  ix=rng.integers(len(Z),size=128);z,ca=b.forward(P,Z[ix]);zz=z-z.max(1,keepdims=True);p=np.exp(zz);p/=p.sum(1,keepdims=True);py=p[np.arange(128),y[ix]];dz=p;dz[np.arange(128),y[ix]]-=1;dz*=py[:,None]**g;dz/=128;G=b.backward(P,ca,dz);o.step(P,G,b.lr_multiplier(t,20000,.02))
 return P,norm
for w in [1,2,3]:
 for g in [.2,.5]:
  t=time.time();P,n=tr(w,g);z=b.forward(P,n(D))[0]; print(w,g,(z.argmax(1)==Y).mean(),round(time.time()-t,1),flush=True)
