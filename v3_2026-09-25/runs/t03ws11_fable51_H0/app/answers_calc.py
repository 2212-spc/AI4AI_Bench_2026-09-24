import numpy as np
from fit import *
import fit4
models={'M1':fit4.M1,'M2':fit4.M2,'M3':fit4.M3,'M5':fit4.M5}
def pred(name,p,Nn,Dd,qq,ss):
    # rebind globals used by model fns
    Nn,Dd,qq,ss=map(lambda x: np.atleast_1d(np.array(x,float)),(Nn,Dd,qq,ss))
    fit4.N,fit4.D,fit4.q,fit4.sub=Nn,Dd,qq,ss
    fit4.U=ss*(1-qq); fit4.ep=Dd/fit4.U; fit4.R=np.maximum(fit4.ep-1,0)
    return models[name](p)
Np=1e9; Dp=2e11
subs=np.linspace(2e10,1e11,81)
for name in models:
    p=np.load(name+'.npy')
    L0=pred(name,p,Np,Dp,0,4e10)[0]; Linf=pred(name,p,Np,Dp,0,1e15)[0]
    q1=L0-Linf
    q2=pred(name,p,Np,Dp,0.5,4e10)[0]-L0
    q3={qq:pred(name,p,Np,Dp,qq,4e10)[0]-L0 for qq in [0,0.3,0.6,0.85]}
    q4=pred(name,p,Np,1e11,0.6,1e15)[0]-pred(name,p,Np,1e11,0,1e15)[0]
    d5=pred(name,p,Np,Dp,0.5,subs)-pred(name,p,Np,Dp,0,subs)
    d6=pred(name,p,Np,Dp,0.6,subs)-pred(name,p,Np,Dp,0,subs)
    print(f"{name}: q1={q1:.4f} q2={q2:+.4f} q3={ {k:round(v,4) for k,v in q3.items()} } q4={q4:+.4f} q5=[{d5.min():+.4f},{d5.max():+.4f}] q6 range=[{d6.min():+.4f},{d6.max():+.4f}]  (q5 at sub=2e10:{d5[0]:+.4f}, 1e11:{d5[-1]:+.4f}; q6 2e10:{d6[0]:+.4f} 1e11:{d6[-1]:+.4f})")
