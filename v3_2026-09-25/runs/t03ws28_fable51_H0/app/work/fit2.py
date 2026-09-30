from data import data
from nm import fit
import numpy as np, sys
N,D,q,sub,seed,L=data.T
U=sub*(1-q)
def deff(D,U,Rs):
    R=np.maximum(D/U-1,0); return U+U*Rs*(1-np.exp(-R/Rs))
forms={
 'sat': lambda q,c,k: 1+c*(1-np.exp(-q/abs(k))),
 'pow': lambda q,c,k: 1+c*q**abs(k),
 'lin': lambda q,c,k: 1+c*q,
 'hill': lambda q,c,k: 1+c*q/(q+abs(k)),
}
def model(p,N,D,q,U,form,floor=False):
    E,lA,a,lB,b,lRs,c,k,e=p
    g=forms[form](q,c,k)
    Deff=deff(D,U,np.exp(lRs))*g
    return E+(e*q if floor else 0)+np.exp(lA)/N**a+np.exp(lB)/Deff**b
pA=np.load('pA.npy')
res={}
for form in forms:
  for floor in [False,True]:
    def f(p):
        if not floor: p=np.r_[p[:8],0]
        r=model(p,N,D,q,U,form,floor)-L; return np.sum(r**2)
    x0=list(pA)+[np.log(5),1.0,0.3]+([0.0] if floor else [])
    p,v=fit(f,x0,restarts=10,iters=40000)
    if not floor: p=np.r_[p[:8],0]
    print(form,'floor' if floor else '','rmse %.4f'%np.sqrt(v/len(L)),np.round(p,3))
    res[(form,floor)]=p
np.save('res2.npy',res,allow_pickle=True)
