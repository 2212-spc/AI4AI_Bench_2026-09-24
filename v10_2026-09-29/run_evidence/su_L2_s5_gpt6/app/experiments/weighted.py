import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json,time
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y']; c0=json.load(open('/app/repo/config.json')); K=10
crow=np.bincount(y,minlength=K)/len(y); gold=np.bincount(Y,minlength=K)/len(Y)
# monkey patch weighted loss by custom train

def train(c,weights=None,seed=0):
 rng=np.random.default_rng(seed); norm=b.standardizer(X);Z=norm(X);P=b.init_params(32,c['hidden'],10,rng);o=b.AdamW(P,c['lr'],c['weight_decay']);bs=c['batch_size']
 for t in range(1000):
  ix=rng.integers(len(Z),size=bs);z,cache=b.forward(P,Z[ix]);zz=z-z.max(1,keepdims=True);p=np.exp(zz);p/=p.sum(1,keepdims=True); dz=p; dz[np.arange(bs),y[ix]]-=1
  if weights is not None: dz*=weights[y[ix]][:,None]
  dz/=bs;G=b.backward(P,cache,dz);o.step(P,G,b.lr_multiplier(t,1000,c['warmup_frac']))
 return P,norm
for mode,w in [('none',None),('gold/crowd',gold/crow),('sqrt',np.sqrt(gold/crow)),('gold',gold),('uniform',1/crow)]:
 for seed in [0]:
  P,n=train(c0,w,seed);z=b.forward(P,n(D))[0]
  print(mode,'raw',float((z.argmax(1)==Y).mean()),'prior powers',end=' ')
  for power in [-.5,0,.25,.5,.75,1]:
   pred=(z+power*np.log(gold/crow)[None,:]).argmax(1);print(power,round((pred==Y).mean(),3),end=';')
  print('trainweights',w)
