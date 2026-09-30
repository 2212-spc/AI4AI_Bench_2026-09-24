import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json,time
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
def run(frac,wd=3,steps=1000,prod=False):
 rng=np.random.default_rng(0);norm=b.standardizer(X);Z=norm(X);P=[];O=[]
 for seed in [0,1]: P.append(b.init_params(32,256,10,np.random.default_rng(seed)));O.append(b.AdamW(P[-1],.003,wd))
 for t in range(steps):
  ix=rng.integers(len(Z),size=128); z=[];ca=[];loss=[]
  for j in range(2):
   zz,cc=b.forward(P[j],Z[ix]);zz-=zz.max(1,keepdims=True);pp=np.exp(zz);pp/=pp.sum(1,keepdims=True);z.append(pp);ca.append(cc);loss.append(-np.log(pp[np.arange(128),y[ix]]+1e-8))
  keep=max(1,int(128*frac));
  for j in range(2):
   sel=np.argpartition(loss[1-j],keep-1)[:keep]; dz=z[j][sel].copy();dz[np.arange(keep),y[ix][sel]]-=1;dz/=keep;G=b.backward(P[j],tuple(v[sel] for v in ca[j]),dz); # backward cache works selected
   O[j].step(P[j],G,b.lr_multiplier(t,steps,.02))
 zz=[]
 for j in range(2):zz.append(b.forward(P[j],norm(D))[0])
 return np.mean(zz,0)
for f in [.5,.6,.7,.8,.9,1]:
 z=run(f);print(f,(z.argmax(1)==Y).mean())
