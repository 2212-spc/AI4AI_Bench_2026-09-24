import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,numpy as np,json
sys.path.insert(0,'/app/work');import base2 as tr
a=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');b=json.load(open('/app/work/original_config.json')); perm=np.random.default_rng(321).permutation(4000)
for n in [500,1000,2000,4000]:
 ix=perm[:n]
 for wd in [.1,.25,.5,1,2,4]:
  for q in [.3,.5,.7]:
   ss=[]
   for seed in [0,1]:
    c=dict(b,weight_decay=wd,loss='gce',gce_q=q);P,no=tr.train(a['X'][ix],a['y'][ix],n//4,c,seed);ss.append(np.mean(tr.predict(P,no,d['X'])==d['y']))
   print(n,wd,q,round(float(np.mean(ss)),4),flush=True)
