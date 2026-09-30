from optimize import *
def jac(fun,p):
 y=fun(p);J=np.empty((y.size,len(p)))
 for j in range(len(p)):
  pp=p.copy();h=1e-5*(1+abs(p[j]));pp[j]+=h;J[:,j]=(fun(pp).ravel()-y.ravel())/h
 return J
if __name__=='__main__':
 p=np.load('full_params.npy');m,l=optimize(p);n,t,u,y=arrays(load());J=jac(lambda p:predict(p,n,t,u),p);C=np.linalg.inv(J.T@J);g=jac(lambda p:np.array([objective(p,m)]),p)[0];v=g@C@g
 out=[]
 for N in [5e7,7.5e7,1e8,1.5e8,2e8,3e8,4e8]:
  for D in [1e9,2e9,4e9,8e9,1e10,2e10]:
   cost=6*N*D
   if cost>8.9e18:continue
   for mix in [[.25]*4,m,[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]:
    tt=np.array([mix])*D/1e9;JJ=jac(lambda p:predict(p,np.array([N/1e8]),tt,U[None,:]),p)
    cg=C@g;red=cg@JJ.T@np.linalg.solve(np.eye(3)+JJ@C@JJ.T,JJ@cg)
    out.append((red/(cost/1e18),red,N,D,list(mix)))
 print('current std',np.sqrt(v)*.004)
 for r in sorted(out,reverse=True)[:20]:print(r)
