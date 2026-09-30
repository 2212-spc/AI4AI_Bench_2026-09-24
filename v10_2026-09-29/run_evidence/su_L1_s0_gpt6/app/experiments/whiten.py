from variants import *
def whitening(X):
 mu=X.mean(0);sd=X.std(0)+1e-6; xs=(X-mu)/sd
 lam,U=np.linalg.eigh(xs.T@xs/len(xs));mat=(U*(np.maximum(lam,1e-4)**-.5))@U.T
 return lambda A: (((A-mu)/sd)@mat).astype(np.float32)
m.standardizer=whitening
seed=int(sys.argv[1])
for noise in [0.,.2,.4]:
 for wd in [.25,.5,1.,2.]:
  c=dict(cfg,noise=noise,weight_decay=wd)
  result=fit(X,y,1000,c,seed)
  print(seed,noise,wd,np.round(result,4),flush=True)
