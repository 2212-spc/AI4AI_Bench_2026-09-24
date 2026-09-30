"""Steady-state M/H2/c+M FIFO queue; type revealed at service start.
Level = number of requests in system; phase = number of B requests in service.
Queued labels remain iid Bernoulli(f), because their patience is type-independent.
"""
import numpy as np

def erlang_a(lam, mean, c, theta):
    lam, mean, c = np.broadcast_arrays(lam,mean,c)
    p=np.ones(lam.shape); total=p.copy(); eq=np.zeros_like(p)
    for n in range(1,350):
        p=p*lam/(np.minimum(n,c)/mean+np.maximum(n-c,0)*theta)
        total+=p; eq+=p*np.maximum(n-c,0)
    wait=eq/total/lam
    return theta*wait,wait

def hyperexp(lam,f,c,theta,sa,sb,qmax=110):
    if f<1e-10 or f>1-1e-10:
        a,w=erlang_a(lam,sa if f<.5 else sb,c,theta)
        return float(a),float(w)
    ma,mb=1/sa,1/sb
    top=c+qmax
    ups=[]; downs=[None]; ds=[]
    for n in range(top+1):
        m=min(n,c);i=np.arange(m+1);a=(m-i)*ma;b=i*mb;q=max(0,n-c)
        ds.append(np.diag(-lam-a-b-q*theta))
        up=np.zeros((m+1,min(n+1,c)+1))
        if n<c:
            up[i,i]=lam*(1-f);up[i,i+1]=lam*f
        else:up[i,i]=lam
        ups.append(up)
        if n>0:
            down=np.zeros((m+1,min(n-1,c)+1))
            if n<=c:
                down[i[:-1],i[:-1]]=a[:-1]
                down[i[1:],i[1:]-1]=b[1:]
            else:
                down[i,i]=a*(1-f)+b*f+q*theta
                down[i[:-1],i[:-1]+1]=a[:-1]*f
                down[i[1:],i[1:]-1]=b[1:]*(1-f)
            downs.append(down)
    rs=[None]*top
    eff=ds[top]
    for n in range(top-1,-1,-1):
        rs[n]=np.linalg.solve(eff.T,-ups[n].T).T
        eff=ds[n]+rs[n]@downs[n+1]
    p=np.ones(1);total=1.;eq=0.
    for n in range(1,top+1):
        p=p@rs[n-1];mass=p.sum();total+=mass;eq+=max(n-c,0)*mass
    wait=eq/total/lam
    return theta*wait,wait

if __name__=='__main__':
    import time
    t=time.time()
    for f in [.3,.5,.8]:
        for lam,c in [(1,6),(2.5,16)]:
            a,w=hyperexp(lam,f,c,1/46,3.075,7.24)
            am,wm=erlang_a(lam,3.075*(1-f)+7.24*f,c,1/46)
            print(lam,f,c,'H2',a,w,'mean approx',am,wm)
    print('seconds',time.time()-t)
