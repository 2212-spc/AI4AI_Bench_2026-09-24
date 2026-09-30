from data import data
from nm import fit
from common import deff
import numpy as np
rng=np.random.default_rng(0)
N,D,q,sub,seed,L=data.T
U=sub*(1-q)
res=np.load('res4.npy',allow_pickle=True).item()
def model(p,N,D,q,U,floor):
    E,lA,a,lB,b,lRs,c,k,e=p
    g=1+c*q**abs(k)
    return E+(e*q if floor else 0)+np.exp(lA)/N**a+np.exp(lB)/(deff(D,U,np.exp(lRs))*g)**b
Np,Dp=1e9,2e11
def stats(p,floor):
    m=lambda qq,s,DD=Dp: model(p,Np,DD,qq,s*(1-qq),floor)
    return [m(0,4e10)-m(0,1e15), m(0.5,4e10)-m(0,4e10), m(0.6,4e10)-m(0.3,4e10), m(0,4e10)-m(0.3,4e10),
            m(0.6,1e15,1e11)-m(0,1e15,1e11), m(0.5,1e11)-m(0,1e11), m(0.5,2e10)-m(0,2e10), m(0.85,2e10)-m(0,2e10), m(0.85,1e11)-m(0,1e11)]
names=['q1','q2','q3:L.6-L.3','q3:L0-L.3','q4','q5lo','q5hi','q6@2e10','q6@1e11']
for floor in [False,True]:
    p0=res[('pow',floor)]
    pred0=model(p0,N,D,q,U,floor); r0=L-pred0
    out=[]
    for bnum in range(40):
        Lb=pred0+rng.choice(r0,len(r0),replace=True)
        def f(p):
            if not floor: p=np.r_[p[:8],0]
            if abs(p[5])>8 or p[6]<-0.99: return 1e9
            return np.sum((model(p,N,D,q,U,floor)-Lb)**2)
        x0=p0[:8] if not floor else p0
        p,v=fit(f,x0,restarts=3,iters=20000,step=0.03)
        if not floor: p=np.r_[p[:8],0]
        out.append(stats(p,floor))
    out=np.array(out)
    print('floor',floor,'point',np.round(stats(p0,floor),4))
    for n,mu,sd,lo,hi in zip(names,out.mean(0),out.std(0),out.min(0),out.max(0)):
        print('  %-10s mean %+.4f sd %.4f  [%+.4f,%+.4f]'%(n,mu,sd,lo,hi))
