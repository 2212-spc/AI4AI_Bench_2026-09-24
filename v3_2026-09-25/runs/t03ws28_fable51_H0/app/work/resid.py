from data import data
from common import deff
import numpy as np
N,D,q,sub,seed,L=data.T
U=sub*(1-q)
res=np.load('res4.npy',allow_pickle=True).item()
p=res[('lin',False)]
def model(p,N,D,q,U):
    E,lA,a,lB,b,lRs,c,k,e=p
    return E+np.exp(lA)/N**a+np.exp(lB)/(deff(D,U,np.exp(lRs))*(1+c*q))**b
pred=model(p,N,D,q,U)
for i in range(len(L)):
    if q[i]>0: print('N=%.1e D=%.1e q=%.2f U=%.2e ep=%5.1f s=%d %.4f pred %.4f res %+.4f'%(N[i],D[i],q[i],U[i],D[i]/U[i],seed[i],L[i],pred[i],L[i]-pred[i]))
