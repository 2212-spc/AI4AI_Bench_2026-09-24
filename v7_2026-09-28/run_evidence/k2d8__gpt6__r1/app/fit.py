import numpy as np,json
DOM=['web','code','math','papers']; EV=['general','code','math']; U=np.array([3e12,36410989526.00656,9967276924.74475,23536988791.987087])/1e9
rows=[json.loads(l) for l in open('/app/runs.jsonl')]
N=np.array([r['N']/5e7 for r in rows]);T=np.array([[r['mix'][d]*r['D']/1e9 for d in DOM] for r in rows]);P=np.array([[r.get('pool',dict(zip(DOM,U*1e9)))[d]/1e9 for d in DOM] for r in rows]);Y=np.array([[r['result']['eval_loss'][e] for e in EV] for r in rows])
def predict(x,N,T,P):
 R=np.exp(x[24:28]);effective=np.minimum(T,P)+P*R*(-np.expm1(-np.maximum(T/P-1,0)/R))
 pars=x[:24].reshape(3,8);c=pars[:,0];a=np.exp(pars[:,1]);alpha=np.exp(pars[:,2]);beta=np.exp(pars[:,3]);w=np.exp(pars[:,4:8])
 return c+a*np.asarray(N)[...,None]**-alpha+(effective@w.T)**-beta

def lm(fun,x,niter=200):
 x=x.copy();lam=.001;r=fun(x);cost=r@r
 for it in range(niter):
  J=np.stack([(fun(x+np.eye(len(x))[i]*1e-5)-r)/1e-5 for i in range(len(x))],axis=1)
  A=J.T@J;g=J.T@r
  step=np.linalg.solve(A+lam*np.diag(np.maximum(np.diag(A),1e-6)), -g)
  xn=x+step
  with np.errstate(over='ignore',invalid='ignore',divide='ignore'):rn=fun(xn);cn=rn@rn
  if np.isfinite(cn) and cn<cost:
   x=xn;r=rn
   if abs(cost-cn)<1e-12:break
   cost=cn;lam=max(lam/3,1e-9)
  else:lam*=5
 return x,cost
if __name__=='__main__':
 x=np.r_[np.tile([1.,np.log(.7),np.log(.3),np.log(.3),0.,0.,0.,0.],3),np.log([20,3,1,3])]
 x,cost=lm(lambda x:(predict(x,N,T,P)-Y).ravel(),x)
 print('rmse',np.sqrt(cost/Y.size),'R',np.exp(x[24:]),'pars',x[:24].reshape(3,8))
 print('residuals',np.round(predict(x,N,T,P)-Y,4))
 np.save('/app/fit.npy',x)
