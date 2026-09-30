from variants import *
seed=int(sys.argv[1])
inds=np.random.default_rng(456+seed).permutation(len(X))
for n in [1000,2000,4000]:
 for wd in [.5,1.,2.,4.]:
  c=dict(cfg,noise=.3,weight_decay=wd)
  result=fit(X[inds[:n]],y[inds[:n]],n//4,c,seed)
  print(seed,n,wd,np.round(result,4),flush=True)
