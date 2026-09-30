import numpy as np
exec(open('fit.py').read().split("np.random.seed(0)")[0])
p6,_=fit('raw',6); r6=model(p6,N,D,F,LR,AR,'raw')-L
# noise per N from residuals
print('resid rms by N:',{n:round(float(np.sqrt(np.mean(r6[N==n]**2))),4) for n in sorted(set(N))})
# residual vs lr bins (cosine only)
mc=np.array([s=='cosine' for s in S])
for lo,hi in [(0.1,0.15),(0.15,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.01)]:
    m=mc&(LR>=lo)&(LR<hi); print('lr[%.2f,%.2f) n=%d mean res %.4f (6p)'%(lo,hi,m.sum(),r6[m].mean()))
# weighted fit with weights 1/sigma_N^2
sig={n:float(np.sqrt(np.mean(r6[N==n]**2))) for n in sorted(set(N))}
W=np.array([1/sig[n]**2 for n in N])
def wfit(npar):
    x0=list(p6)+([1.0] if npar==7 else [])
    f=lambda p: np.sum(W*(model(p,N,D,F,LR,AR,'raw')-L)**2)/np.sum(W)
    best=None
    for t in range(3):
        p,v=nm(f,x0 if best is None else best[0]*(1+0.03*np.random.randn(npar)))
        if best is None or v<best[1]: best=(p,v)
    return best
for npar in [6,7]:
    p,v=wfit(npar); print('weighted',npar,np.round(p,4),'chi2/n %.3f'%v)
# alternative: penalty linear but D_eff = D*(f - k*(1-lr)) i.e. lr-dependent lag
def model2(p,N,D,F,LR,AR):
    E,lA,a,lB,b,P,k=p
    Deff=D*np.maximum(F-k*(1-LR)*F,1e-3)
    return E+np.exp(lA)*N**-a+np.exp(lB)*Deff**-b+P*LR
f=lambda p: np.mean((model2(p,N,D,F,LR,AR)-L)**2)
p,v=nm(f,list(p6)+[0.0]); print('lag model',np.round(p,4),'rmse %.4f'%np.sqrt(v))
