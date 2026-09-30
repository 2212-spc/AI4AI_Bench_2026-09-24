import numpy as np, json
from lab import load
R=[r for r in load() if r["batch"]==2048]
T=np.array([r["tokens"] for r in R]); X=np.log([r["lr"] for r in R]); Z=np.log([r["wd"] for r in R])
Y=np.array([r["result"]["final_val_loss"] for r in R]); N=np.array([r["seeds"] for r in R],float)
hs=sorted(set(T)); hi={h:i for i,h in enumerate(hs)}; H=np.array([hi[t] for t in T])
lt=np.log(T)
def model(p, lt, X, Z, H=None, L0=None):
    a=np.exp(p[0]); al,be=p[1],p[2]; ga,de=p[3],p[4]; ka,la=p[5],p[6]; rho=p[7]; cub=p[8]
    xs=al+be*lt; vs=ga+de*lt; c=np.exp(ka+la*lt)
    d=X-xs
    base=a*d**2+cub*d**3+c*(Z+rho*X-vs)**2
    if L0 is None: return base
    return L0[H]+base
def nm(f,x0,steps=4000,scale=0.3):
    n=len(x0); S=[np.array(x0,float)]
    for i in range(n):
        x=np.array(x0,float); x[i]+=scale; S.append(x)
    S=np.array(S); F=np.array([f(s) for s in S])
    for it in range(steps):
        o=np.argsort(F); S=S[o]; F=F[o]
        cen=S[:-1].mean(0); xr=cen+(cen-S[-1]); fr=f(xr)
        if fr<F[0]:
            xe=cen+2*(cen-S[-1]); fe=f(xe)
            if fe<fr: S[-1]=xe;F[-1]=fe
            else: S[-1]=xr;F[-1]=fr
        elif fr<F[-2]: S[-1]=xr;F[-1]=fr
        else:
            xc=cen+0.5*(S[-1]-cen); fc=f(xc)
            if fc<F[-1]: S[-1]=xc;F[-1]=fc
            else:
                S[1:]=S[0]+0.5*(S[1:]-S[0]); F[1:]=[f(s) for s in S[1:]]
    o=np.argsort(F); return S[o][0],F[o][0]
def loss(p):
    base=model(p,lt,X,Z)
    # profile out L0 per horizon (weighted mean of residual)
    res=Y-base; L0=np.array([np.sum(N[H==i]*res[H==i])/np.sum(N[H==i]) for i in range(len(hs))])
    r=res-L0[H]
    return np.sum(N*r**2)
p0=[np.log(0.07), np.log(0.0011)+0.5*np.log(6), -0.5, np.log(0.01)-0.5*np.log(6)+0*0, 0.5, np.log(0.002), 0.5, 0.0, 0.0]
best=None
for trial in range(6):
    x0=np.array(p0)+np.random.RandomState(trial).normal(0,0.1,len(p0))*(trial>0)
    p,f=nm(loss,x0,6000)
    if best is None or f<best[1]: best=(p,f)
p,f=best
n=len(Y); k=len(p)+len(hs)
print("rms",np.sqrt(f/(n-k)),"n",n)
a=np.exp(p[0]); print("a_lr",a,"cubic",p[8],"rho",p[7])
print("lr* exponent",p[2],"  wd-term exponent(d)",p[4]," c exponent",p[6])
for t in hs+[192,384]:
    l=np.log(t); xs=p[1]+p[2]*l; vs=p[3]+p[4]*l; c=np.exp(p[5]+p[6]*l)
    # optimum: minimize a d^2 + cub d^3 + c(z+rho x - vs)^2 -> d=0 (approx), z=vs-rho*xs
    print(f"t={t:6g} lr*={np.exp(xs):.6f} wd*={np.exp(vs-p[7]*xs):.5f} c={c:.4f} lr*wd={np.exp(xs)*np.exp(vs-p[7]*xs):.2e}")
np.save("gfit_p.npy",p)
# residuals
base=model(p,lt,X,Z); res=Y-base; L0=np.array([np.sum(N[H==i]*res[H==i])/np.sum(N[H==i]) for i in range(len(hs))])
for i,r in enumerate(R):
    print(f'{r["tokens"]:5g} lr={r["lr"]:.6f} wd={r["wd"]:.4f} y={Y[i]:.4f} resid={res[i]-L0[H[i]]:+.4f}')
