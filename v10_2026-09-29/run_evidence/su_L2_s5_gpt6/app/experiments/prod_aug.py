import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json,time
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');rng=np.random.default_rng(123);X=np.tile(s['X'],(20,1));y=np.tile(s['y'],20);q=rng.permutation(len(y));X=X[q];y=y[q];D,Y=d['X'],d['y'];base=json.load(open('/app/repo/config.json'))
for sig in [.02,.05,.1,.2,.3,.5,1.0]:
 for w in [1,2,3]:
  c=dict(base,weight_decay=w,aug_sigma=sig);t=time.time();P,n=b.train(X,y,20000,c,seed=0);p=b.predict(P,n,D);print(sig,w,float((p==Y).mean()),np.round(np.bincount(p,minlength=10)/len(p),3),round(time.time()-t,1),flush=True)
