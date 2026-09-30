import numpy as np, json
def nm(f, x0, iters=20000, step=0.1):
    # Nelder-Mead
    n=len(x0); pts=[np.array(x0,float)]
    for i in range(n):
        p=np.array(x0,float); p[i]+= step*(abs(p[i]) if p[i]!=0 else 1); pts.append(p)
    vals=[f(p) for p in pts]
    for it in range(iters):
        o=np.argsort(vals); pts=[pts[i] for i in o]; vals=[vals[i] for i in o]
        c=np.mean(pts[:-1],axis=0); xr=c+(c-pts[-1]); fr=f(xr)
        if fr<vals[0]:
            xe=c+2*(c-pts[-1]); fe=f(xe)
            if fe<fr: pts[-1],vals[-1]=xe,fe
            else: pts[-1],vals[-1]=xr,fr
        elif fr<vals[-2]: pts[-1],vals[-1]=xr,fr
        else:
            xc=c+0.5*(pts[-1]-c); fc=f(xc)
            if fc<vals[-1]: pts[-1],vals[-1]=xc,fc
            else:
                for i in range(1,len(pts)):
                    pts[i]=pts[0]+0.5*(pts[i]-pts[0]); vals[i]=f(pts[i])
    o=np.argsort(vals); return pts[o[0]], vals[o[0]]

# fresh data q=0 runs
fresh=[(1.5e7,3e8,6.3366),(3e7,6e8,5.4907),(6e7,1.2e9,4.8085),(1.2e8,2.4e9,4.2527),(2.4e8,4.8e9,3.8118),
       (5e7,1e9,4.9735),(5e7,4e9,4.4665),(5e7,1.6e10,4.1176)]
N=np.array([r[0] for r in fresh]); D=np.array([r[1] for r in fresh]); L=np.array([r[2] for r in fresh])
def model(p,N,D):
    E,lA,a,lB,b=p
    return E+np.exp(lA)*N**(-a)+np.exp(lB)*D**(-b)
def obj(p): return np.sum((model(p,N,D)-L)**2)
best=None
for a0 in [0.2,0.3,0.5]:
  for b0 in [0.2,0.3,0.5]:
    p,v=nm(obj,[1.5,np.log(1e3),a0,np.log(1e3),b0],step=0.3)
    p,v=nm(obj,p,step=0.05)
    if best is None or v<best[1]: best=(p,v)
p,v=best
print("fresh fit",p,"rss",v,"resid",model(p,N,D)-L)
np.save('/app/p_fresh.npy',p)
