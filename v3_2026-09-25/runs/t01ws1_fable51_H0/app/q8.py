import numpy as np
exec(open('/app/fit2.py').read().split("print(\"model B")[0])
L0=betaB[:K]; N=np.array([k[0] for k in keys]); D=np.array([k[1] for k in keys])
def fitL(L0,N,D):
    def sse(t):
        al,be=t
        X=np.c_[np.ones(len(L0)),N**-al,D**-be]
        b=np.linalg.lstsq(X,L0,rcond=None)[0]; r=L0-X@b; return (r*r).sum(),b
    best=None
    for a0 in [0.2,0.35,0.5]:
        for b0 in [0.2,0.35,0.5]:
            t,v=nelder_mead(lambda t: sse(t)[0],[a0,b0],step=0.05,iters=3000)
            if best is None or v<best[1]: best=(t,v)
    t=best[0]; v,b=sse(t); return t,b,v
t,b,v=fitL(L0,N,D)
print("alpha=%.3f beta=%.3f E=%.3f A=%.2f B=%.2f rms=%.4f"%(t[0],t[1],b[0],b[1],b[2],np.sqrt(v/(K-5))))
pred=lambda t,b,N,D: b[0]+b[1]*N**-t[0]+b[2]*D**-t[1]
print("q8 point: %.4f"%pred(t,b,1e9,1e12))
print("check at (1e9,2e10): %.4f, (1e9,2e11): %.4f"%(pred(t,b,1e9,2e10),pred(t,b,1e9,2e11)))
X=np.c_[np.ones(K),N**-t[0],D**-t[1]]; fit=X@b
rng=np.random.default_rng(1); vals=[]
for i in range(200):
    Lb=fit+rng.normal(0,0.008,K)
    tb,bb,_=fitL(Lb,N,D); vals.append(pred(tb,bb,1e9,1e12))
vals=np.array(vals); print("boot mean %.4f sd %.4f  5-95%%: %.4f %.4f"%(vals.mean(),vals.std(),*np.percentile(vals,[5,95])))
# Alternative: fit using raw loss values via joint model A L0's
