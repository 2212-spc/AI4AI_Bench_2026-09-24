import numpy as np
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
