import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json,time
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];base=json.load(open('/app/repo/config.json'))
for h in [32,64,128,256,512]:
 for lr in [.001,.003,.01]:
  for wd in [2,3,5]:
   c=dict(base,hidden=h,lr=lr,weight_decay=wd);P,n=b.train(X,y,1000,c,seed=0);p=b.predict(P,n,D);print(h,lr,wd,round((p==Y).mean(),3),flush=True)
