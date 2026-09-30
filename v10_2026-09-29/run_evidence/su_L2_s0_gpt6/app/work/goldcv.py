import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,numpy as np,json
sys.path.insert(0,'/app/work');import original_train as tr
D=np.load('/app/data/dev.npz');X,y=D['X'],D['y']; base=json.load(open('/app/work/original_config.json')); rng=np.random.default_rng(4)
for h in [32,64,128,256]:
 for wd in [.01,.1,.5,1,2]:
  ss=[]
  for rep in range(3):
   ix=rng.permutation(len(y)); c=dict(base,hidden=h,weight_decay=wd);P,n=tr.train(X[ix[:800]],y[ix[:800]],1000,c,rep);ss.append(np.mean(tr.predict(P,n,X[ix[800:]])==y[ix[800:]]))
  print(h,wd,np.mean(ss),ss,flush=True)
