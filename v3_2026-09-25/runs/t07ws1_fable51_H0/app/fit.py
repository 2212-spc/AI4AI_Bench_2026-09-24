import numpy as np, json
from math import pi, sin, cos
rows=np.load('rows.npy',allow_pickle=True)
def lr(s,f):
    if s=='cosine': return 0.1+0.45*(1+cos(pi*f))
    return 1.0 if f<=0.8 else (1-f)/0.2
def area(s,f):
    if s=='cosine': return 0.1*f+0.45*(f+sin(pi*f)/pi)
    return f if f<=0.8 else 0.8+(f-0.8)-(f-0.8)**2/0.4
N=np.array([r[0] for r in rows],float); D=np.array([r[1] for r in rows],float)
S=[r[2] for r in rows]; F=np.array([r[3] for r in rows],float); L=np.array([r[5] for r in rows],float)
LR=np.array([lr(s,f) for s,f in zip(S,F)]); AR=np.array([area(s,f) for s,f in zip(S,F)])
def nm(fun,x0,iters=20000,step=0.3):
    n=len(x0); pts=[np.array(x0,float)]
    for i in range(n):
        p=np.array(x0,float); p[i]+=step*(abs(p[i]) if p[i]!=0 else 1); pts.append(p)
    vals=[fun(p) for p in pts]
    for it in range(iters):
        o=np.argsort(vals); pts=[pts[i] for i in o]; vals=[vals[i] for i in o]
        c=np.mean(pts[:-1],axis=0); xr=c+(c-pts[-1]); fr=fun(xr)
        if fr<vals[0]:
            xe=c+2*(c-pts[-1]); fe=fun(xe)
            if fe<fr: pts[-1],vals[-1]=xe,fe
            else: pts[-1],vals[-1]=xr,fr
        elif fr<vals[-2]: pts[-1],vals[-1]=xr,fr
        else:
            xc=c+0.5*(pts[-1]-c); fc=fun(xc)
            if fc<vals[-1]: pts[-1],vals[-1]=xc,fc
            else:
                for i in range(1,len(pts)): pts[i]=pts[0]+0.5*(pts[i]-pts[0]); vals[i]=fun(pts[i])
        if max(vals)-min(vals)<1e-12 and it>2000: break
    return pts[0],vals[0]
def model(p,N,D,F,LR,AR,mode):
    E,lA,a,lB,b,P=p[:6]
    Deff = D*F if mode=='raw' else D*AR
    pen = P*LR if len(p)==6 else P*LR**p[6]
    return E+np.exp(lA)*N**-a+np.exp(lB)*Deff**-b+pen
def fit(mode,npar=6,mask=None):
    m=np.ones(len(L),bool) if mask is None else mask
    x0=[1.5,np.log(50),0.4,np.log(500),0.35,0.1]+([1.0] if npar==7 else [])
    f=lambda p: np.mean((model(p,N[m],D[m],F[m],LR[m],AR[m],mode)-L[m])**2)
    best=None
    for tr in range(3):
        p,v=nm(f,x0 if best is None else best[0]*(1+0.05*np.random.randn(npar)))
        if best is None or v<best[1]: best=(p,v)
    return best
np.random.seed(0)
for mode in ['raw','area']:
    for npar in [6,7]:
        p,v=fit(mode,npar); print(mode,npar,'rmse',np.sqrt(v),'params',np.round(p,4))
        res=model(p,N,D,F,LR,AR,mode)-L
        # residual by schedule/f
        for s in ['cosine','wsd']:
            ms=np.array([x==s for x in S])
            print('  ',s,'rmse',np.sqrt(np.mean(res[ms]**2)))
print('---- residuals by run, raw 6')
p,v=fit('raw',6)
res=model(p,N,D,F,LR,AR,'raw')-L
keys=sorted(set(zip(N,D,S)))
for k in keys:
    m=(N==k[0])&(D==k[1])&np.array([s==k[2] for s in S])
    print(k, 'n',m.sum(),'mean res %.4f rmse %.4f'%(res[m].mean(),np.sqrt(np.mean(res[m]**2))), 'fracs',np.round(F[m],2)[:6], np.round(res[m],3)[:8])
# fit P per-N separately (penalty by N) using fixed E,A,a,B,b
print('---- per-N penalty estimate (fixing others)')
for n0 in sorted(set(N)):
    m=(N==n0)
    base=p[0]+np.exp(p[1])*n0**-p[2]+np.exp(p[3])*(D[m]*F[m])**-p[4]
    x=LR[m]; y=L[m]-base
    print(n0, 'P=%.4f'%(np.sum(x*y)/np.sum(x*x)), 'n',m.sum())
