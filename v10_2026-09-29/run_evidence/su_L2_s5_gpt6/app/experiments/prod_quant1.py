import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json,time
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');rng=np.random.default_rng(123);X=np.tile(s['X'],(20,1));y=np.tile(s['y'],20);q=rng.permutation(len(y));X=X[q];y=y[q];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
for typ in ['quant']:
 lo=np.quantile(X,.01,0);hi=np.quantile(X,.99,0);mu=X.mean(0);sd=X.std(0)+1e-6; qm=(lo+hi)/2;qs=(hi-lo)/2
 def f(A):return ((A-mu)/sd if typ=='std' else np.clip((A-qm)/qs,-3,3)).astype('float32')
 old=b.standardizer;b.standardizer=lambda a:(lambda q:f(q))
 for w in [2]:
  t=time.time();P,n=b.train(X,y,20000,dict(c,weight_decay=w),seed=0);p=b.predict(P,n,D);print(typ,w,(p==Y).mean(),round(time.time()-t,1),flush=True)
 b.standardizer=old
