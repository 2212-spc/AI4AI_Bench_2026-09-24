import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,numpy as np,json,time
sys.path.insert(0,'/app/work');import original_train as tr
A=np.load('/app/data/sample.npz');D=np.load('/app/data/dev.npz');b=json.load(open('/app/work/original_config.json'));rng=np.random.default_rng(88);X=A['X'];y=A['y'];
for sig in [.001,.01,.05,.1]:
 xx=np.tile(X,(20,1))+rng.normal(0,sig*X.std(0),(80000,32)).astype('float32'); yy=np.tile(y,20)
 for wd in [.05,.1,.2,.4]:
  vals=[]
  for seed in [0,1]:
   c=dict(b,weight_decay=wd);P,n=tr.train(xx,yy,20000,c,seed);vals.append(np.mean(tr.predict(P,n,D['X'])==D['y']))
  print(sig,wd,vals,np.mean(vals),flush=True)
