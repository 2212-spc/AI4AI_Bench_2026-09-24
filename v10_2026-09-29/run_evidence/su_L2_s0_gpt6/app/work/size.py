import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,numpy as np,json,time
sys.path.insert(0,'/app/work');import original_train as tr
a=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');b=json.load(open('/app/work/original_config.json'))
for h in [384,512,768,1024]:
 for wd in [1,2,3]:
  ss=[]
  for seed in [0,1]:
   c=dict(b,hidden=h,weight_decay=wd);P,n=tr.train(a['X'],a['y'],1000,c,seed);ss.append(np.mean(tr.predict(P,n,d['X'])==d['y']))
  print(h,wd,ss,np.mean(ss),flush=True)
