import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
for sm in [0,.02,.05,.1,.2,.3,.5,.7,.9]:
 for wd in [1,2,3,4,5]:
  cc=dict(c,label_smoothing=sm,weight_decay=wd);P,n=b.train(X,y,1000,cc,seed=0);z=b.forward(P,n(D))[0]; print(sm,wd,round((z.argmax(1)==Y).mean(),3),flush=True)
