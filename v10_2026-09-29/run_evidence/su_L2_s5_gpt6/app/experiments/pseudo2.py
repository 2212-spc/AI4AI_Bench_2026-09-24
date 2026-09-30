import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
def tr(th,burn):
 rng=np.random.default_rng(0);norm=b.standardizer(X);Z=norm(X);P=b.init_params(32,256,10,rng);o=b.AdamW(P,.003,3)
 for t in range(1000):
  ix=rng.integers(len(Z),size=128);z,ca=b.forward(P,Z[ix]);zz=z-z.max(1,keepdims=True);p=np.exp(zz);p/=p.sum(1,keepdims=True); yy=y[ix].copy(); pred=p.argmax(1); pc=p[np.arange(128),pred]; bad=(pred!=yy)&(pc>th)
  if t>=burn: yy[bad]=pred[bad]
  dz=p;dz[np.arange(128),yy]-=1;dz/=128;G=b.backward(P,ca,dz);o.step(P,G,b.lr_multiplier(t,1000,.02))
 return b.predict(P,norm,D)
for burn in [200,400,600]:
 for th in [.5,.6,.7,.8,.9,.95]: print(burn,th,round((tr(th,burn)==Y).mean(),3),flush=True)
