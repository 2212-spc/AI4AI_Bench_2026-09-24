import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import numpy as np, sys, json, time
sys.path.insert(0,'/app/experiments')
import baseline as m
s=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz')
X,y=s['X'],s['y']; D,Y=d['X'],d['y']
cfg=json.load(open('/app/repo/config.json'))
results=[]
for n in [500,1000,2000,4000]:
 ids=np.random.default_rng(43).permutation(len(X))[:n]
 for wd in [0.15,0.5,1.0,2.0,3.0,5.0,8.0]:
  c=dict(cfg,weight_decay=wd); t=time.process_time()
  P,norm=m.train(X[ids],y[ids],n//4,c,seed=0)
  p=m.predict(P,norm,D)
  r=dict(n=n,wd=wd,dev=float((p==Y).mean()),train=float((m.predict(P,norm,X[ids])==y[ids]).mean()),cpu=round(time.process_time()-t,3))
  results.append(r); print(r,flush=True)
json.dump(results,open('/app/experiments/scaling.json','w'),indent=2)
print('linear ridge classifiers')
Z=(X-X.mean(0))/X.std(0); Z=np.column_stack([Z,np.ones(len(Z))]); Zd=np.column_stack([(D-X.mean(0))/X.std(0),np.ones(len(D))]); T=np.eye(10)[y]
for ridge in [0.1,10,100,1000]:
 W=np.linalg.solve(Z.T@Z+ridge*np.eye(Z.shape[1]),Z.T@T); print(ridge,(np.argmax(Zd@W,1)==Y).mean())
print('feature means',X.mean(0)); print('feature std',X.std(0)); print('correlation eigenvalues',np.linalg.eigvalsh(np.corrcoef(X.T)))
