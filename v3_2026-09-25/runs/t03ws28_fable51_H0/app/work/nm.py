import numpy as np
def nelder_mead(f, x0, step=0.1, iters=20000, tol=1e-12):
    n=len(x0); pts=[np.array(x0,float)]
    for i in range(n):
        p=np.array(x0,float); p[i]+= step*(abs(p[i]) if p[i]!=0 else 1); pts.append(p)
    vals=[f(p) for p in pts]
    for it in range(iters):
        o=np.argsort(vals); pts=[pts[i] for i in o]; vals=[vals[i] for i in o]
        if abs(vals[-1]-vals[0])<tol and it>200: break
        c=np.mean(pts[:-1],axis=0)
        xr=c+(c-pts[-1]); fr=f(xr)
        if fr<vals[0]:
            xe=c+2*(c-pts[-1]); fe=f(xe)
            if fe<fr: pts[-1],vals[-1]=xe,fe
            else: pts[-1],vals[-1]=xr,fr
        elif fr<vals[-2]: pts[-1],vals[-1]=xr,fr
        else:
            xc=c+0.5*(pts[-1]-c); fc=f(xc)
            if fc<vals[-1]: pts[-1],vals[-1]=xc,fc
            else:
                for i in range(1,n+1):
                    pts[i]=pts[0]+0.5*(pts[i]-pts[0]); vals[i]=f(pts[i])
    o=np.argmin(vals); return pts[o],vals[o]
def fit(f,x0,restarts=5,**kw):
    best=None
    x=np.array(x0,float)
    for r in range(restarts):
        x,v=nelder_mead(f,x,**kw)
        if best is None or v<best[1]: best=(x,v)
    return best
