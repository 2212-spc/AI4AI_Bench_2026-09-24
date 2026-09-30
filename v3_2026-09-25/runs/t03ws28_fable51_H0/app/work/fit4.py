from data import data
from nm import fit
from common import deff
import numpy as np
N,D,q,sub,seed,L=data.T
U=sub*(1-q)
prep=np.load('prep.npy')
forms={
 'sat': lambda q,c,k: 1+c*(1-np.exp(-q/abs(k))),
 'pow': lambda q,c,k: 1+c*q**abs(k),
 'lin': lambda q,c,k: 1+c*q,
 'hill': lambda q,c,k: 1+c*q/(q+abs(k)),
 'quad': lambda q,c,k: 1+c*q+k*q*q,
}
def model(p,N,D,q,U,form):
    E,lA,a,lB,b,lRs,c,k,e=p
    g=forms[form](q,c,k)
    return E+e*q+np.exp(lA)/N**a+np.exp(lB)/(deff(D,U,np.exp(lRs))*g)**b
res={}
for form in forms:
  for floor in [False,True]:
    def f(p):
        if not floor: p=np.r_[p[:8],0]
        if abs(p[5])>8 or (form!='quad' and p[6]<-0.99): return 1e9
        g=forms[form](q,p[6],p[7])
        if np.any(g<=0.01): return 1e9
        r=model(p,N,D,q,U,form)-L; return np.sum(r**2)
    best=None
    for c0,k0 in [(1,0.3),(2,0.5),(0.5,1),(3,0.2)]:
        x0=list(prep)+[c0,k0]+([0.0] if floor else [])
        p,v=fit(f,x0,restarts=6,iters=40000,step=0.05)
        if best is None or v<best[1]: best=(p,v)
    p,v=best
    if not floor: p=np.r_[p[:8],0]
    print('%-5s %-5s rmse %.4f'%(form,'floor' if floor else '',np.sqrt(v/len(L))),np.round(p,3))
    res[(form,floor)]=p
np.save('res4.npy',res,allow_pickle=True)
