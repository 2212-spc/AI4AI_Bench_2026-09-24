import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'));crow=np.bincount(y,minlength=10)/len(y); gold=np.bincount(Y,minlength=10)/len(Y)
def tr(w,exp):
 rng=np.random.default_rng(0);norm=b.standardizer(X);Z=norm(X);P=b.init_params(32,256,10,rng);o=b.AdamW(P,.003,w);weights=(gold/crow)**exp
 for t in range(1000):
  ix=rng.integers(len(Z),size=128);z,ca=b.forward(P,Z[ix]);zz=z-z.max(1,keepdims=True);p=np.exp(zz);p/=p.sum(1,keepdims=True);dz=p;dz[np.arange(128),y[ix]]-=1;dz*=weights[y[ix]][:,None];dz/=128;G=b.backward(P,ca,dz);o.step(P,G,b.lr_multiplier(t,1000,.02))
 return b.predict(P,norm,D)
for e in [-1,-.75,-.5,-.25,-.1,0,.1,.25,.5,.75,1]:
 for w in [2,3]: print(e,w,round((tr(w,e)==Y).mean(),3),flush=True)
