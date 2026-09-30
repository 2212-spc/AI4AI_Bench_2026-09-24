import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json,time
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');
# Make a larger iid-like training set from the available uniform sample; shuffle copies to match row sampling.
rng=np.random.default_rng(123); X=np.tile(s['X'],(20,1)); y=np.tile(s['y'],20); q=rng.permutation(len(y));X=X[q];y=y[q]
D,Y=d['X'],d['y']; base=json.load(open('/app/repo/config.json'))
cs=[dict(base,weight_decay=w) for w in [2,3,4,5,6]]+[dict(base,weight_decay=3,label_smoothing=.05),dict(base,weight_decay=3,aug_sigma=.03),dict(base,weight_decay=3,hidden=128),dict(base,weight_decay=3,hidden=512)]
for c in cs:
 t=time.time();P,n=b.train(X,y,20000,c,seed=0);p=b.predict(P,n,D);print(c,'dev',float((p==Y).mean()),'pred',np.bincount(p,minlength=10)/len(p),'sec',round(time.time()-t,1),flush=True)
