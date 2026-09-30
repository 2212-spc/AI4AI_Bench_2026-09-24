import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,numpy as np,json,time
sys.path.insert(0,'/app/work');import base2 as tr
a=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');b=json.load(open('/app/work/original_config.json'))
for mode,par in [('gce',.7),('gce',.9),('bootstrap',.2),('bootstrap',.5),('bootstrap',.8),('bootstrap',.95)]:
 vals=[]
 for seed in [0,1]:
  c=dict(b,weight_decay=2,loss=mode,**({'gce_q':par} if mode=='gce' else {'bootstrap_beta':par})); P,n=tr.train(a['X'],a['y'],1000,c,seed); vals.append(np.mean(tr.predict(P,n,d['X'])==d['y']))
 print(mode,par,vals,np.mean(vals),flush=True)
