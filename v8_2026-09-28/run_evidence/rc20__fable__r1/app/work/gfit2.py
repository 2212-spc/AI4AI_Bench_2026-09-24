import numpy as np, sys
from lab import load
from gfit import nm
tmin=float(sys.argv[1]) if len(sys.argv)>1 else 6
R=[r for r in load() if r["batch"]==2048 and r["tokens"]>=tmin]
T=np.array([r["tokens"] for r in R]); X=np.log([r["lr"] for r in R]); Z=np.log([r["wd"] for r in R])
Y=np.array([r["result"]["final_val_loss"] for r in R]); N=np.array([r["seeds"] for r in R],float)
hs=sorted(set(T)); hi={h:i for i,h in enumerate(hs)}; H=np.array([hi[t] for t in T]); lt=np.log(T)
def base(p,lt,X,Z,rho_fix=None):
    a=np.exp(p[0]); xs=p[1]+p[2]*lt; vs=p[3]+p[4]*lt; c=np.exp(p[5]+p[6]*lt); rho=p[7] if rho_fix is None else rho_fix; cub=p[8]
    d=X-xs; return a*d**2+cub*d**3+c*(Z+rho*X-vs)**2
def fit(rho_fix=None):
    def loss(p):
        b=base(p,lt,X,Z,rho_fix); res=Y-b
        L0=np.array([np.sum(N[H==i]*res[H==i])/np.sum(N[H==i]) for i in range(len(hs))])
        return np.sum(N*(res-L0[H])**2)
    p0=[np.log(0.075), np.log(0.0011)+0.55*np.log(6), -0.55, np.log(0.01)+0.4*np.log(6)+0.74*np.log(0.0011)+0.74*0.55*np.log(6)*0, 0.0, np.log(0.0012)-1.2*np.log(6), 1.2, 0.74, 0.0]
    best=None
    for trial in range(3):
        x0=np.array(p0)+np.random.RandomState(trial).normal(0,0.15,len(p0))*(trial>0)
        p,f=nm(loss,x0,4000)
        if best is None or f<best[1]: best=(p,f)
    return best
for rf in (None,0.0,1.0):
    p,f=fit(rf); k=len(p)-(rf is not None)+len(hs)
    print("rho_fix",rf,"rms",np.sqrt(f/(len(Y)-k)),"a",np.exp(p[0]),"cub",p[8],"rho",p[7] if rf is None else rf,"lr exp",p[2],"c exp",p[6])
    rho=p[7] if rf is None else rf
    for t in [48,96,192,384]:
        l=np.log(t); xs=p[1]+p[2]*l; vs=p[3]+p[4]*l; c=np.exp(p[5]+p[6]*l)
        print(f"   t={t:4g} lr*={np.exp(xs):.6f} wd*={np.exp(vs-rho*xs):.5f} c={c:.4f}")
    if rf is None: np.save("gfit2_p.npy",p)
