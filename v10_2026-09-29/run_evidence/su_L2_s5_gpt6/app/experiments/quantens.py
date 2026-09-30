import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'));lo=np.quantile(X,.01,0);hi=np.quantile(X,.99,0);mu=(lo+hi)/2;sd=(hi-lo)/2
old=b.standardizer;b.standardizer=lambda a:(lambda q:np.clip((q-mu)/sd,-3,3).astype('float32'))
# lda on same transformed
A=b.standardizer(X)(X);Q=b.standardizer(X)(D);M=np.array([A[y==k].mean(0) for k in range(10)]);C=(A-M[y]).T@(A-M[y])/len(A);ci=np.linalg.inv(C+.001*np.eye(32));l=Q@ci@M.T-.5*np.sum(M*(M@ci),1);l=(l-l.mean(1)[:,None])/(l.std(1)[:,None]+1e-6)
for wd in [2,3]:
 P,n=b.train(X,y,1000,dict(c,weight_decay=wd),seed=0);z=b.forward(P,n(D))[0];z=(z-z.mean(1)[:,None])/(z.std(1)[:,None]+1e-6); print(wd,[(a,round(((z+a*l).argmax(1)==Y).mean(),3)) for a in [0,.05,.1,.15,.2,.25,.3,.4]])
