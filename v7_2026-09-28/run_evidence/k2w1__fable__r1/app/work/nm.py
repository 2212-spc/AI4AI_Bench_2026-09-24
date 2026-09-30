import numpy as np
def nelder_mead(f, x0, step=0.1, iters=4000, tol=1e-10):
    x0=np.asarray(x0,float); n=len(x0)
    pts=[x0]+[x0+np.eye(n)[i]*(step if np.isscalar(step) else step[i]) for i in range(n)]
    vals=[f(p) for p in pts]
    for _ in range(iters):
        o=np.argsort(vals); pts=[pts[i] for i in o]; vals=[vals[i] for i in o]
        if abs(vals[-1]-vals[0])<tol and np.max(np.abs(np.array(pts[-1])-pts[0]))<1e-8: break
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
                pts=[pts[0]]+[pts[0]+0.5*(p-pts[0]) for p in pts[1:]]; vals=[vals[0]]+[f(p) for p in pts[1:]]
    i=int(np.argmin(vals)); return pts[i],vals[i]
def fit(f,x0,restarts=5,**kw):
    best=None
    for r in range(restarts):
        x,v=nelder_mead(f,x0 if r==0 else best[0]*(1+0.05*np.random.randn(len(x0))),**kw)
        if best is None or v<best[1]: best=(x,v)
    return best
