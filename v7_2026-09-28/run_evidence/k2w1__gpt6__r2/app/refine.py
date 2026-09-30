from design import *
from experiment import run
p=np.load('full_params.npy');budget=load()[-1]['budget_left']
while budget>=3e17:
 m,l=optimize(p);n,t,u,y=arrays(load());J=jac(lambda p:predict(p,n,t,u),p);C=np.linalg.inv(J.T@J);g=jac(lambda p:np.array([objective(p,m)]),p)[0];cg=C@g
 best=None
 for N in [5e7,7.5e7,1e8,1.5e8,2e8,3e8,4e8]:
  for D in [1e9,2e9,4e9,8e9,1e10,2e10]:
   cost=6*N*D
   if cost>budget:continue
   for mix in [[.25]*4,m,[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]:
    tt=np.array([mix])*D/1e9;JJ=jac(lambda p:predict(p,np.array([N/1e8]),tt,U[None,:]),p)
    red=cg@JJ.T@np.linalg.solve(np.eye(3)+JJ@C@JJ.T,JJ@cg)
    score=red/(cost/1e18)
    if best is None or score>best[0]:best=(score,N,D,mix)
 if best is None:break
 _,N,D,mix=best
 r=run(mix,N=N,D=D);budget=r['budget_left'];p,v=fullfit(start=p);np.save('full_params.npy',p)
 print('forecast',objective(p,m),'std approx',.004*np.sqrt(g@C@g),'rms',np.sqrt(v/(len(load())*3)),flush=True)
