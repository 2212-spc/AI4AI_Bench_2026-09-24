import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json,time
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c0=json.load(open('/app/repo/config.json'))
def tr(c,mode,seed=0,steps=1000):
 rng=np.random.default_rng(seed);norm=b.standardizer(X);Z=norm(X);P=b.init_params(32,c['hidden'],10,rng);o=b.AdamW(P,c['lr'],c['weight_decay']);bs=128
 for t in range(steps):
  ix=rng.integers(len(Z),size=bs);z,ca=b.forward(P,Z[ix]);zz=z-z.max(1,keepdims=True);p=np.exp(zz);p/=p.sum(1,keepdims=True);py=p[np.arange(bs),y[ix]]; dz=p.copy();dz[np.arange(bs),y[ix]]-=1
  if mode[0]=='gce': dz*=py[:,None]**mode[1]
  if mode[0]=='focal': dz*=((1-py)**mode[1])[:,None] # approximate focal
  if mode[0]=='clip': dz*= (py>mode[1])[:,None]
  if mode[0]=='mix':
   # blend hard labels with predictions; low-confidence gradient reduced
   dz*=((1-mode[1])*py+mode[1])[:,None]
  dz/=bs;G=b.backward(P,ca,dz);o.step(P,G,b.lr_multiplier(t,steps,c['warmup_frac']))
 return P,norm
for typ,vals in [('gce',[.1,.2,.3,.5,.7,1]),('focal',[.2,.5,1,2]),('clip',[.05,.1,.2,.3]),('mix',[.1,.2,.4,.6,.8])]:
 for v in vals:
  for wd in [2,3,4]:
   P,n=tr(dict(c0,weight_decay=wd),(typ,v));p=b.predict(P,n,D);print(typ,v,wd,round((p==Y).mean(),3),flush=True)
