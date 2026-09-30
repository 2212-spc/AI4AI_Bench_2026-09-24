import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,numpy as np,json
sys.path.insert(0,'/app/work');import original_train as tr
a=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');b=json.load(open('/app/work/original_config.json'))
for lr in [.0003,.001,.003,.01,.03]:
 for wd in [.5,1,2,3,4]:
  vals=[]
  for seed in [0,1]:
   c=dict(b,lr=lr,weight_decay=wd);P,n=tr.train(a['X'],a['y'],1000,c,seed);vals.append(np.mean(tr.predict(P,n,d['X'])==d['y']))
  print(lr,wd,vals,np.mean(vals),flush=True)
