import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json,time
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');rng=np.random.default_rng(123);X=np.tile(s['X'],(20,1));y=np.tile(s['y'],20);q=rng.permutation(len(y));X=X[q];y=y[q];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'));crow=np.bincount(y,minlength=10)/len(y); gold=np.bincount(Y,minlength=10)/len(Y)
for wd in [2,3,4,5]:
 P,n=b.train(X,y,20000,dict(c,weight_decay=wd),seed=0);z=b.forward(P,n(D))[0];print('wd',wd,'raw',(z.argmax(1)==Y).mean(),end=' ')
 for p in [-.5,-.25,0,.1,.2,.3,.4,.5,.7,1]:
  pred=(z+p*np.log(gold/crow)).argmax(1);print(p,round((pred==Y).mean(),3),end=';')
 print(flush=True)
