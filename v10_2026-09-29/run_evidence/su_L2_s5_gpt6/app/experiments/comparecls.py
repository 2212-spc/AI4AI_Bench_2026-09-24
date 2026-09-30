# rerun standard and gce and print class acc / confusion
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments');import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
def gce():
 rng=np.random.default_rng(0);norm=b.standardizer(X);Z=norm(X);P=b.init_params(32,256,10,rng);o=b.AdamW(P,.003,2)
 for t in range(1000):
  ix=rng.integers(len(Z),size=128);z,ca=b.forward(P,Z[ix]);z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True);py=p[np.arange(128),y[ix]];dz=p;dz[np.arange(128),y[ix]]-=1;dz*=py[:,None]**.5;dz/=128;o.step(P, b.backward(P,ca,dz),b.lr_multiplier(t,1000,.02))
 return b.predict(P,norm,D)
for name,p in [('base',b.predict(*b.train(X,y,1000,dict(c,weight_decay=3),0),D)),('gce',gce())]: print(name,(p==Y).mean(),[round(((p==Y)&(Y==k)).sum()/max(1,(Y==k).sum()),2) for k in range(10)],np.bincount(p,minlength=10))
