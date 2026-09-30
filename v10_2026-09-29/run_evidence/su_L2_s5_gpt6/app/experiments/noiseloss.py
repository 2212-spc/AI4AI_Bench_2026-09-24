import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
def tr(e,w=3,kind='forward',steps=1000):
 rng=np.random.default_rng(0);norm=b.standardizer(X);Z=norm(X);P=b.init_params(32,256,10,rng);o=b.AdamW(P,.003,w);a=1-e;bb=e/9
 for t in range(steps):
  ix=rng.integers(len(Z),size=128);z,ca=b.forward(P,Z[ix]);z-=z.max(1,keepdims=True);q=np.exp(z);q/=q.sum(1,keepdims=True)
  if kind=='forward':
   r=bb+a*q; py=r[np.arange(128),y[ix]]; dz=(a*q*(1-py[:,None]))/py[:,None] # derivative? d loss/d logits = a q_j*(I?); actual dL/dq y=-a/py; softmax grad = -a/py*q*(onehot-q)
   dz=(a*q[np.arange(128),y[ix]]/py)[:,None]*q;dz[np.arange(128),y[ix]]-=a*q[np.arange(128),y[ix]]/py
  elif kind=='backward':
   # inverse target, clip target
   r=(q[np.arange(128),y[ix]]-bb)/(a-bb); dz=q; dz[np.arange(128),y[ix]]-=1; dz*=np.clip(r,0,1)[:,None] # bad approx
  dz/=128;G=b.backward(P,ca,dz);o.step(P,G,b.lr_multiplier(t,steps,.02))
 return b.predict(P,norm,D)
for kind in ['forward']:
 for e in [.1,.2,.3,.4,.5,.6,.7]:
  for w in [1,2,3]:print(kind,e,w,round((tr(e,w,kind)==Y).mean(),3),flush=True)
