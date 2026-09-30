import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
mu=X.mean(0);sd=X.std(0)+1e-6;a=(X-mu)/sd;q=(D-mu)/sd;M=np.array([a[y==k].mean(0) for k in range(10)]);C=(a-M[y]).T@(a-M[y])/len(a);ci=np.linalg.inv(C+.001*np.eye(32));l=q@ci@M.T-.5*np.sum(M*(M@ci),1);ll=(l-l.mean(1)[:,None])/(l.std(1)[:,None]+1e-6)
rng=np.random.default_rng(0);norm=b.standardizer(X);Z=norm(X);P=b.init_params(32,256,10,rng);o=b.AdamW(P,.003,2)
for t in range(1000):
 ix=rng.integers(len(Z),size=128);z,ca=b.forward(P,Z[ix]);zz=z-z.max(1,keepdims=True);p=np.exp(zz);p/=p.sum(1,keepdims=True);py=p[np.arange(128),y[ix]];dz=p;dz[np.arange(128),y[ix]]-=1;dz*=py[:,None]**.5;dz/=128;G=b.backward(P,ca,dz);o.step(P,G,b.lr_multiplier(t,1000,.02))
z=b.forward(P,norm(D))[0];zz=(z-z.mean(1)[:,None])/(z.std(1)[:,None]+1e-6)
print('raw',(z.argmax(1)==Y).mean())
for a in [0,.1,.2,.25,.3,.4,.5,.7,1]: print(a,((zz+a*ll).argmax(1)==Y).mean())
