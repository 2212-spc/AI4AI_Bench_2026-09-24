import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json,time
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');rng=np.random.default_rng(123);X=np.tile(s['X'],(20,1));y=np.tile(s['y'],20);q=rng.permutation(len(y));X=X[q];y=y[q];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
for lr in [.0005,.001,.002,.003,.005,.01]:
 for w in [3,4,5,6]:
  cc=dict(c,lr=lr,weight_decay=w);t=time.time();P,n=b.train(X,y,20000,cc,seed=0);p=b.predict(P,n,D);print(lr,w,round((p==Y).mean(),3),round(time.time()-t,1),flush=True)
