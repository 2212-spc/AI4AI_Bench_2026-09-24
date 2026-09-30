from common import deff
import numpy as np
res=np.load('res4.npy',allow_pickle=True).item()
forms={
 'sat': lambda q,c,k: 1+c*(1-np.exp(-q/abs(k))),
 'pow': lambda q,c,k: 1+c*q**abs(k),
 'lin': lambda q,c,k: 1+c*q,
 'hill': lambda q,c,k: 1+c*q/(q+abs(k)),
 'quad': lambda q,c,k: 1+c*q+k*q*q,
}
def model(p,form,N,D,q,sub):
    E,lA,a,lB,b,lRs,c,k,e=p
    U=sub*(1-q); g=forms[form](q,c,k)
    return E+e*q+np.exp(lA)/N**a+np.exp(lB)/(deff(D,U,np.exp(lRs))*g)**b
Np,Dp=1e9,2e11
print('form   floor  q1      q2      q3(0,.3,.6,.85)              q4     q5lo    q5hi    q6(2e10,1e11)')
for (form,floor),p in res.items():
    q1=model(p,form,Np,Dp,0,4e10)-model(p,form,Np,Dp,0,1e15)
    q2=model(p,form,Np,Dp,0.5,4e10)-model(p,form,Np,Dp,0,4e10)
    q3=[model(p,form,Np,Dp,qq,4e10) for qq in (0,0.3,0.6,0.85)]
    q4=model(p,form,Np,1e11,0.6,1e15)-model(p,form,Np,1e11,0,1e15)
    q5=[model(p,form,Np,Dp,0.5,s)-model(p,form,Np,Dp,0,s) for s in np.linspace(2e10,1e11,81)]
    q6=[model(p,form,Np,Dp,0.85,s)-model(p,form,Np,Dp,0,s) for s in (2e10,4e10,1e11)]
    print('%-5s %-5s %.4f %+.4f %s %+.4f %+.4f %+.4f %s'%(form,floor,q1,q2,np.round(q3,4),q4,min(q5),max(q5),np.round(q6,4)))
