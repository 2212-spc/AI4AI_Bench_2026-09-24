import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
mu=X.mean(0);sd=X.std(0)+1e-6;a=(X-mu)/sd;q=(D-mu)/sd;M=np.array([a[y==k].mean(0) for k in range(10)]);C=(a-M[y]).T@(a-M[y])/len(a);ci=np.linalg.inv(C+.001*np.eye(32));l=q@ci@M.T-.5*np.sum(M*(M@ci),1);ll=(l-l.mean(1)[:,None])/(l.std(1)[:,None]+1e-6)
for wd in [2,3,4,5]:
 for seed in [0,1,2,3]:
  P,n=b.train(X,y,1000,dict(c,weight_decay=wd),seed=seed);z=b.forward(P,n(D))[0];zz=(z-z.mean(1)[:,None])/(z.std(1)[:,None]+1e-6); print(wd,seed,round((z.argmax(1)==Y).mean(),3),[round(((zz+a*ll).argmax(1)==Y).mean(),3) for a in [.1,.2,.25,.3,.4]],flush=True)
