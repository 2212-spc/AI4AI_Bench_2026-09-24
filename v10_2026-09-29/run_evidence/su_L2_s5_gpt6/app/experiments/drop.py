import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
def tr(drop,w):
 rng=np.random.default_rng(0);norm=b.standardizer(X);Z=norm(X);P=b.init_params(32,256,10,rng);o=b.AdamW(P,.003,w)
 for t in range(1000):
  ix=rng.integers(len(Z),size=128); xb=Z[ix];z,ca=b.forward(P,xb); # apply masks to cached h and corresponding backprop via dz? 
  # manually masks act outputs, backward modified
  x,a1,h1,a2,h2=ca;m1=(rng.random(h1.shape)>drop).astype('float32')/(1-drop);m2=(rng.random(h2.shape)>drop).astype('float32')/(1-drop);h1*=m1; a2=h1@P['W2']+P['b2'];h2=np.maximum(a2,0)*m2;z=h2@P['W3']+P['b3'];ca=(x,a1,h1,a2,h2);zz=z-z.max(1,keepdims=True);p=np.exp(zz);p/=p.sum(1,keepdims=True);dz=p;dz[np.arange(128),y[ix]]-=1;dz/=128
  G=b.backward(P,ca,dz);o.step(P,G,b.lr_multiplier(t,1000,.02))
 return b.predict(P,norm,D)
for dr in [.0,.05,.1,.2,.3,.5]:
 for w in [2,3,4]:print(dr,w,round((tr(dr,w)==Y).mean(),3),flush=True)
