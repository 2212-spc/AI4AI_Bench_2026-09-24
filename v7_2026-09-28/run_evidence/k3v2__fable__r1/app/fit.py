import numpy as np
data = [(256,1000,5.17),(256,4000,4.79),(256,16000,4.588),
        (512,1000,4.94),(512,4000,4.40),(512,16000,4.093),
        (1024,1000,5.065),(1024,4000,4.259)]
n=np.array([d[0] for d in data],float); S=np.array([d[1] for d in data],float); L=np.array([d[2] for d in data])
def model(p,n,S):
    E,A,a,B,b,c = p
    return E + A*n**(-a) + B*(n**c)*S**(-b)
def loss(p): return np.sum((model(p,n,S)-L)**2)
# crude random-restart Nelder-Mead
def nm(f,x0,iters=4000):
    from itertools import count
    N=len(x0); simplex=[np.array(x0,float)]
    for i in range(N):
        x=np.array(x0,float); x[i]+= 0.1*(abs(x[i])+0.1); simplex.append(x)
    fs=[f(x) for x in simplex]
    for it in range(iters):
        idx=np.argsort(fs); simplex=[simplex[i] for i in idx]; fs=[fs[i] for i in idx]
        c=np.mean(simplex[:-1],axis=0); xr=c+(c-simplex[-1]); fr=f(xr)
        if fr<fs[0]:
            xe=c+2*(c-simplex[-1]); fe=f(xe)
            if fe<fr: simplex[-1],fs[-1]=xe,fe
            else: simplex[-1],fs[-1]=xr,fr
        elif fr<fs[-2]: simplex[-1],fs[-1]=xr,fr
        else:
            xc=c+0.5*(simplex[-1]-c); fc=f(xc)
            if fc<fs[-1]: simplex[-1],fs[-1]=xc,fc
            else:
                for i in range(1,len(simplex)):
                    simplex[i]=simplex[0]+0.5*(simplex[i]-simplex[0]); fs[i]=f(simplex[i])
    i=np.argmin(fs); return simplex[i],fs[i]
rng=np.random.default_rng(0)
best=None
for t in range(60):
    x0=[rng.uniform(1,4), rng.uniform(1,50), rng.uniform(0.2,1.0), rng.uniform(1,50), rng.uniform(0.2,0.8), rng.uniform(0,1.5)]
    x,fv=nm(loss,x0)
    if best is None or fv<best[1]: best=(x,fv)
p,fv=best
print("params E,A,a,B,b,c =",p, "rss",fv, "rms", np.sqrt(fv/len(L)))
print("resid", model(p,n,S)-L)
C=1.00663296e12
for w in [724,1024,1448,2048,2896,4096,5793,8192]:
    St=C/w**2; print(w, int(St), round(model(p,w,St),3))
np.save('/app/params.npy',p)

print("--- ensemble of acceptable fits ---")
rng=np.random.default_rng(1)
acc=[]
for t in range(300):
    x0=[rng.uniform(1,4), rng.uniform(1,50), rng.uniform(0.2,1.0), rng.uniform(1,50), rng.uniform(0.2,0.8), rng.uniform(0,1.5)]
    x,fv=nm(loss,x0,iters=2500)
    if fv<2.0e-4: acc.append((x,fv))
print(len(acc))
ws=[1024,1448,2048,2896,4096]
for x,fv in acc[:40]:
    print(np.round(x,3), "%.1e"%fv, [round(model(x,w,C/w**2),3) for w in ws])
