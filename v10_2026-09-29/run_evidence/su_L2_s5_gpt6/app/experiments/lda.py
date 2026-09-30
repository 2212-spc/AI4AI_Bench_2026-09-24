import numpy as np
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y']; K=10
mu=X.mean(0);sd=X.std(0);z=(X-mu)/sd;q=(D-mu)/sd
for trim in [0,0.01,0.03,0.05,0.1]:
 means=[]
 for k in range(K):
  a=z[y==k]; c=a.mean(0)
  if trim:
   # robust iterative reweight / trim farthest points
   for _ in range(3):
    dist=((a-c)**2).sum(1); a=a[dist<=np.quantile(dist,1-trim)];c=a.mean(0)
  means.append(c)
 means=np.array(means); C=(z-means[y]).T@(z-means[y])/len(z)
 for reg in [.001,.003,.01,.02,.03,.05,.1]:
  Ci=np.linalg.inv(C+reg*np.eye(32)); base=q@Ci@means.T-.5*np.sum(means*(means@Ci),1)
  print('trim',trim,'reg',reg,end=' ')
  for power in [-2,-1,-.5,0,.5,1,2]:
   prior=(np.bincount(y,minlength=K)/len(y))**power; pred=(base+np.log(prior)[None,:]).argmax(1); print(power,round((pred==Y).mean(),3),end=';')
  print()
