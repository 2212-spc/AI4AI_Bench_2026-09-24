from data import data
from nm import fit
import numpy as np
N,D,q,sub,seed,L=data.T
U=sub*(1-q)
fresh=(q==0)&(D<=U)
# Model A: L=E+A/N^a+B/D^b on fresh q=0
def mA(p,N,D):
    E,lA,a,lB,b=p; return E+np.exp(lA)/N**a+np.exp(lB)/D**b
def lossA(p):
    r=mA(p,N[fresh],D[fresh])-L[fresh]; return np.sum(r**2)
p,v=fit(lossA,[2.0,np.log(50),0.3,np.log(50),0.3],restarts=8,iters=40000)
print('fresh fit',p,'rmse',np.sqrt(v/fresh.sum()))
for i in np.where(fresh)[0]: print('%.1e %.1e %.4f pred %.4f'%(N[i],D[i],L[i],mA(p,N[i],D[i])))
np.save('pA.npy',p)
