from variants import *
def whitening(X):
 mu=X.mean(0);sd=X.std(0)+1e-6; xs=(X-mu)/sd
 lam,U=np.linalg.eigh(xs.T@xs/len(xs));mat=(U*(np.maximum(lam,1e-4)**-.5))@U.T
 return lambda A: (((A-mu)/sd)@mat).astype(np.float32)
m.standardizer=whitening
seed=int(sys.argv[1]);inds=np.random.default_rng(456+seed).permutation(len(X))
for n in [1000,2000,4000]:
 for wd in [1.,2.,4.,8.]:
  c=dict(cfg,noise=.25,weight_decay=wd)
  result=fit(X[inds[:n]],y[inds[:n]],n//4,c,seed)
  print(seed,n,wd,np.round(result,4),flush=True)
