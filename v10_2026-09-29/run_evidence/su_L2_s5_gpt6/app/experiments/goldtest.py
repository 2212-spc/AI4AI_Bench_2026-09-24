import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
D=np.load('/app/data/dev.npz');X,y=D['X'],D['y']; c=json.load(open('/app/repo/config.json'))
for n in [500,800]:
 for wd in [.5,1,2,3,5]:
  tr=np.arange(n);te=np.arange(n,1000); cc=dict(c,weight_decay=wd);P,no=b.train(X[tr],y[tr],n//4,cc,seed=0);p=b.predict(P,no,X[te]);print(n,wd,(p==y[te]).mean())
