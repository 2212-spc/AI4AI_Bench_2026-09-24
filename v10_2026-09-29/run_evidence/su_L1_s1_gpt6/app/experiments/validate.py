import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,numpy as np
sys.path.insert(0,'/app/repo')
import train as tr
r=np.random.default_rng(7);z=r.normal(size=(5,4));y=r.integers(4,size=5)
for noise in [0.,.25]:
 for smooth in [0.,.1]:
  loss,g=tr.softmax_xent_grad(z,y,4,smooth,noise);numeric=np.empty_like(z)
  for idx in np.ndindex(z.shape):
   zp=z.copy();zm=z.copy();zp[idx]+=1e-5;zm[idx]-=1e-5
   numeric[idx]=(tr.softmax_xent_grad(zp,y,4,smooth,noise)[0]-tr.softmax_xent_grad(zm,y,4,smooth,noise)[0])/2e-5
  print('gradient',noise,smooth,np.max(np.abs(g-numeric)))
  assert np.allclose(g,numeric,atol=1e-7)
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');cfg=json.load(open('/app/repo/config.json'))
for seed in [0,1]:
 p=tr.train_predict(s['X'],s['y'],1000,cfg,d['X'],seed=seed)
 print('sample ensemble seed',seed,'accuracy',np.mean(p==d['y']),flush=True)
 assert p.shape==d['y'].shape and p.dtype==np.int64
np.save('/app/experiments/dev_X.npy',d['X'])
# Repetition is only for resource measurement; it does not add statistical data.
np.savez('/app/experiments/timing_corpus.npz',X=np.tile(s['X'],(20,1)),y=np.tile(s['y'],20))
np.save('/app/experiments/timing_test.npy',np.tile(d['X'],(20,1)))
