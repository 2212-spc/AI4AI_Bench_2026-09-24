import numpy as np
exec(open('/app/analysis/smooth.py').read().split('for exp in')[0])
ids=[]
for idx,r in enumerate(runs):ids.extend([idx]*(1+len(r.get('checkpoints') or [])+len(r.get('cooldown_branches') or [])))
ids=np.array(ids);assert len(ids)==len(y)
for tau in [0,.5,1,2,4]:
 w=(n/1e8)**.25
 # invert relative covariance I+tau**2 ZZT before N size weights
 V=np.eye(len(n))
 for rid in set(ids):
  inds=np.where(ids==rid)[0];V[np.ix_(inds,inds)]-=tau**2/(1+tau**2*len(inds))
 V*=w[:,None]*w[None,:]
 p=np.array([1.5,.7,.35,.94,.35,.11,.7]);lam=1e-6
 for it in range(1000):
  z=f(p);r=z-y
  J=np.column_stack([(f(p+np.eye(7)[j]*1e-6)-z)/1e-6 for j in range(7)])
  st=np.linalg.solve(J.T@V@J+lam*np.eye(7),-J.T@V@r);nr=f(p+st)-y
  if nr@V@nr<r@V@r:p+=st;lam*=.8
  else:lam*=2
  if np.linalg.norm(st)<1e-10:break
 E,A,al,B,be,K,ga=p
 base=lambda N,D:E+A*(N/1e8)**-al+B*(D/1e9)**-be
 print('tau',tau,'params',p,'q2',base(3e9,6e10))
 print('run res',[tuple(np.round([np.mean((f(p)-y)[ids==i]),np.std((f(p)-y)[ids==i])],4)) for i in set(ids)])
