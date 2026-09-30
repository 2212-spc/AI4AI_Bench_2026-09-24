import numpy as np
def nelder_mead(f, x0, step=None, maxiter=20000, tol=1e-12):
    x0=np.asarray(x0,float); d=len(x0)
    if step is None: step=np.where(x0!=0, 0.2*np.abs(x0), 0.1)
    pts=[x0]+[x0+np.eye(d)[i]*step[i] for i in range(d)]
    vals=[f(p) for p in pts]
    for it in range(maxiter):
        idx=np.argsort(vals); pts=[pts[i] for i in idx]; vals=[vals[i] for i in idx]
        if abs(vals[-1]-vals[0])<tol and it>100: break
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
                pts=[pts[0]]+[pts[0]+0.5*(p-pts[0]) for p in pts[1:]]
                vals=[vals[0]]+[f(p) for p in pts[1:]]
    i=int(np.argmin(vals)); return pts[i],vals[i]
def multi(f,x0s,restarts=3,**kw):
    best=None
    for x0 in x0s:
        x=np.array(x0,float)
        for _ in range(restarts):
            x,v=nelder_mead(f,x,**kw)
        if best is None or v<best[1]: best=(x,v)
    return best
