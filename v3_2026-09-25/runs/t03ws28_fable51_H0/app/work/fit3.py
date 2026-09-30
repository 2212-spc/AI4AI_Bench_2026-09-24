from data import data
from nm import fit
import numpy as np
N,D,q,sub,seed,L=data.T
U=sub*(1-q)
pA=np.load('pA.npy')
from common import deff
rep=(q==0)
def mrep(p,N,D,U):
    E,lA,a,lB,b,lRs=p
    return E+np.exp(lA)/N**a+np.exp(lB)/deff(D,U,np.exp(lRs))**b
# stage 1: only Rs
best=None
for lRs in np.linspace(-2,5,141):
    p=np.r_[pA,lRs]; v=np.sum((mrep(p,N[rep],D[rep],U[rep])-L[rep])**2)
    if best is None or v<best[1]: best=(lRs,v)
print('stage1 Rs=%.3f rmse %.4f'%(np.exp(best[0]),np.sqrt(best[1]/rep.sum())))
def f(p):
    if abs(p[5])>8: return 1e9
    r=mrep(p,N[rep],D[rep],U[rep])-L[rep]; return np.sum(r**2)
p,v=fit(f,list(pA)+[best[0]],restarts=10,iters=40000,step=0.02)
print('rep fit',np.round(p,4),'Rs=%.3f'%np.exp(p[5]),'rmse %.4f'%np.sqrt(v/rep.sum()))
for i in np.where(rep)[0]: print('%.1e D=%.1e U=%.2e ep=%5.1f %.4f pred %.4f'%(N[i],D[i],U[i],D[i]/U[i],L[i],mrep(p,N[i],D[i],U[i])))
np.save('prep.npy',p)
