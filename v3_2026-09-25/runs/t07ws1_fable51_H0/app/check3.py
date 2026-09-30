import numpy as np
from math import pi,sin,cos
exec(open('fit.py').read().split("np.random.seed(0)")[0])
np.random.seed(2)
def lrc(f): return 0.1+0.45*(1+cos(pi*f))
def lrw(f): return 1.0 if f<=0.8 else (1-f)/0.2
n=len(L)
def base(p,N,D,F,AR,k=0.0):
    E,lA,a,lB,b=p[:5]; Deff=D*((1-k)*F+k*AR)
    return E+np.exp(lA)*N**-a+np.exp(lB)*Deff**-b, Deff
variants={
 'lin':        (6, lambda p,N,D,F,LR,AR: base(p,N,D,F,AR)[0]+p[5]*LR),
 'pow':        (7, lambda p,N,D,F,LR,AR: base(p,N,D,F,AR)[0]+p[5]*np.abs(LR)**p[6]),
 'lin+area':   (7, lambda p,N,D,F,LR,AR: base(p,N,D,F,AR,p[6])[0]+p[5]*LR),
 'lin*Deff^-c':(7, lambda p,N,D,F,LR,AR: base(p,N,D,F,AR)[0]+p[5]*LR*(base(p,N,D,F,AR)[1]/1e9)**-p[6]),
 'lin*N^-c':   (7, lambda p,N,D,F,LR,AR: base(p,N,D,F,AR)[0]+p[5]*LR*(N/1e8)**-p[6]),
 'lin*(L-E)':  (7, lambda p,N,D,F,LR,AR: base(p,N,D,F,AR)[0]+p[5]*LR*(1+p[6]*(base(p,N,D,F,AR)[0]-p[0]))),
 'sat':        (7, lambda p,N,D,F,LR,AR: base(p,N,D,F,AR)[0]+p[5]*LR/(1+p[6]*LR)),
}
p6=[1.5033,5.6547,0.3254,7.3104,0.3567,0.1093]
def predv(fn,p,N_,D_,f,s):
    l=lrc(f) if s=='cosine' else lrw(f)
    return fn(p,np.array([N_]),np.array([D_]),np.array([f]),np.array([l]),np.array([area(s,f)]))[0]
for name,(npar,fn) in variants.items():
    x0=p6 if npar==6 else p6+([1.0] if name=="pow" else [0.0])
    if name=='lin+area': x0=p6+[0.1]
    f=lambda p: np.mean((fn(p,N,D,F,LR,AR)-L)**2)
    best=None
    for t in range(4):
        p,v=nm(f,x0 if best is None else best[0]+0.02*np.random.randn(npar)*(np.abs(best[0])+0.1))
        if best is None or v<best[1]: best=(p,v)
    p,v=best
    P=lambda N_,D_,fr,s: predv(fn,p,N_,D_,fr,s)
    q2=P(3e9,6e10,1,'wsd'); q3=P(3e9,6e10,0.4,'cosine')-P(3e9,2.4e10,1,'wsd')
    fin=P(3e9,6e10,1,'cosine'); g=[P(3e9,6e10,fr,'cosine')-fin for fr in np.linspace(0.5,0.9,41)]
    q5=P(3e9,6e10,0.5,'cosine')-P(3e9,0.35*6e10,1,'wsd')
    q1={k:round(P(nn,1.08e21/(6*nn),1,'cosine'),4) for k,nn in zip('ABCD',[2.7e8,5.3e8,1.1e9,2.1e9])}
    print('%-12s rmse %.5f'%(name,np.sqrt(v)),np.round(p,3))
    print('    q2 %.4f q3 %.4f q4 [%.4f,%.4f] q5 %.4f'%(q2,q3,min(g),max(g),q5),q1)
