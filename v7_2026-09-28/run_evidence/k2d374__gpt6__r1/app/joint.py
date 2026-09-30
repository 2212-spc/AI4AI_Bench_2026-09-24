from fit import *
U0=np.array([3e12,12947831710.681202,11184250724.25906,10934306213.632713])/1e9
X=np.array([[r['mix'][d]*r['D']/1e9 for d in DOM] for r in data]);N=np.array([r['N']/5e7 for r in data]);Y=np.array([[r['eval_loss'][e] for e in EV] for r in data]);U=np.array([[r.get('pool',dict(zip(DOM,U0*1e9)))[d]/1e9 for d in DOM] for r in data])
def effective(X,U,R):
 return np.minimum(X,U)+U*R*(-np.expm1(-np.maximum(X/U-1,0)/R))
def pred(p,X,N,U):
 Z=effective(X,U,np.exp(p[24:]));return np.column_stack([predict(p[e*8:(e+1)*8],Z,N) for e in range(3)])
def optim(fun,x):
 # Nelder Mead
 simplex=np.array([x]+[x+np.eye(len(x))[i]*.3 for i in range(len(x))]);f=np.array([fun(s) for s in simplex])
 for _ in range(1000):
  order=np.argsort(f);simplex=simplex[order];f=f[order]
  if np.max(np.abs(simplex[1:]-simplex[0]))<1e-8:break
  center=simplex[:-1].mean(axis=0);r=2*center-simplex[-1];fr=fun(r)
  if fr<f[0]:
   ex=3*center-2*simplex[-1];fe=fun(ex)
   simplex[-1],f[-1]=(ex,fe) if fe<fr else (r,fr)
  elif fr<f[-2]:simplex[-1]=r;f[-1]=fr
  else:
   co=center+.5*((r if fr<f[-1] else simplex[-1])-center);fc=fun(co)
   if fc<min(fr,f[-1]):simplex[-1]=co;f[-1]=fc
   else:
    simplex[1:]=(simplex[1:]+simplex[0])/2;f[1:]=[fun(s) for s in simplex[1:]]
 return simplex[0],f[0]
def mix(z):
 z=np.r_[z,0];w=np.exp(z-z.max());return w/w.sum()
if __name__=='__main__':
 p0=np.r_[np.load('/app/base.npy').ravel(),np.log([15,2,1.3,1.9])]
 p,c=lm(lambda p:(pred(p,X,N,U)-Y).ravel(),p0,400)
 print('RMSE',np.sqrt(c/Y.size),'R',np.exp(p[24:]),'params',p[:24].reshape(3,8))
 np.save('/app/joint.npy',p)
 residual=pred(p,X,N,U)-Y
 for j in np.argsort(np.linalg.norm(residual,axis=1))[-8:]:print(data[j]['call_id'],residual[j])
 fun=lambda z:float(pred(p,(500*mix(z))[None,:],np.array([50]),U0[None,:])@np.array([.121,.256,.623]))
 z,f=optim(fun,np.zeros(3));print('opt',mix(z),f,'eval',pred(p,(500*mix(z))[None,:],np.array([50]),U0[None,:]))
