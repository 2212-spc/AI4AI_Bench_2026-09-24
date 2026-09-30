import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'));P,n=b.train(X,y,1000,c,seed=0); z=b.forward(P,n(D))[0]
# LDA variants
mu=X.mean(0);sd=X.std(0);a=(X-mu)/sd;q=(D-mu)/sd;M=np.array([a[y==k].mean(0) for k in range(10)]);C=(a-M[y]).T@(a-M[y])/len(a)
for reg in [.0001,.001,.003,.01,.03,.1]:
 ci=np.linalg.inv(C+reg*np.eye(32));l=q@ci@M.T-.5*np.sum(M*(M@ci),1);print('lda',reg,(l.argmax(1)==Y).mean(),end=' ')
 for alpha in [-2,-1,-.5,0,.25,.5,.75,1,1.5,2,3]:
  # standardized logits and blend
  zz=(z-z.mean(1)[:,None])/(z.std(1)[:,None]+1e-6); ll=(l-l.mean(1)[:,None])/(l.std(1)[:,None]+1e-6)
  print(alpha,round(((zz+alpha*ll).argmax(1)==Y).mean(),3),end=';')
 print()
