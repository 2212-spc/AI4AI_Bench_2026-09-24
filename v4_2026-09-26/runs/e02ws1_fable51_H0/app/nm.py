import numpy as np
def nelder_mead(f, x0, step=0.1, iters=4000, tol=1e-12):
    x0=np.array(x0,float); n=len(x0)
    simplex=[x0]+[x0+np.eye(n)[i]*step*max(abs(x0[i]),1e-3) for i in range(n)]
    vals=[f(x) for x in simplex]
    for it in range(iters):
        order=np.argsort(vals); simplex=[simplex[i] for i in order]; vals=[vals[i] for i in order]
        if abs(vals[-1]-vals[0])<tol*(abs(vals[0])+1e-30) and it>200: break
        c=np.mean(simplex[:-1],axis=0)
        xr=c+(c-simplex[-1]); fr=f(xr)
        if fr<vals[0]:
            xe=c+2*(c-simplex[-1]); fe=f(xe)
            if fe<fr: simplex[-1],vals[-1]=xe,fe
            else: simplex[-1],vals[-1]=xr,fr
        elif fr<vals[-2]: simplex[-1],vals[-1]=xr,fr
        else:
            xc=c+0.5*(simplex[-1]-c); fc=f(xc)
            if fc<vals[-1]: simplex[-1],vals[-1]=xc,fc
            else:
                for i in range(1,n+1):
                    simplex[i]=simplex[0]+0.5*(simplex[i]-simplex[0]); vals[i]=f(simplex[i])
    i=int(np.argmin(vals)); return simplex[i],vals[i]
