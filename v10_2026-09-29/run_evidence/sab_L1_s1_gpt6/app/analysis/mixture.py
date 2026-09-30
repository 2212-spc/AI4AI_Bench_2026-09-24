"""Exact continuous-time Markov model for M/H2/c+M, used as a sensitivity check."""
import numpy as np

def mixture(lam,f,ga,gb,c,theta,qmax=90,tol=2e-12):
    # State (total in system, B requests in service); queued labels are iid.
    states=[(n,j) for n in range(c+qmax+1) for j in range(min(n,c)+1)]
    lookup={s:i for i,s in enumerate(states)}
    src=[];dst=[];rate=[]
    def add(i,s,r):
        if r>0:
            src.append(i);dst.append(lookup[s]);rate.append(r)
    for i,(n,j) in enumerate(states):
        busy=min(n,c);na=busy-j
        if n<c:
            add(i,(n+1,j),lam*(1-f));add(i,(n+1,j+1),lam*f)
        elif n<c+qmax:add(i,(n+1,j),lam)
        if n<=c:
            if na:add(i,(n-1,j),na/ga)
            if j:add(i,(n-1,j-1),j/gb)
        else:
            add(i,(n-1,j),(n-c)*theta)
            if na:
                add(i,(n-1,j),na/ga*(1-f));add(i,(n-1,j+1),na/ga*f)
            if j:
                add(i,(n-1,j-1),j/gb*(1-f));add(i,(n-1,j),j/gb*f)
    src=np.array(src);dst=np.array(dst);rate=np.array(rate)
    outgoing=np.bincount(src,weights=rate,minlength=len(states)); uni=outgoing.max()
    rate/=uni;stay=1-outgoing/uni
    prob=np.zeros(len(states));prob[0]=1
    for it in range(30000):
        new=prob*stay+np.bincount(dst,weights=prob[src]*rate,minlength=len(states))
        if np.max(np.abs(new-prob))<tol:break
        prob=new
    nq=np.array([max(n-c,0) for n,j in states]);w=np.dot(new,nq)/lam
    return theta*w,w,it

if __name__=='__main__':
    from model import erlang
    for lam,f,c in [(1,.3,7),(2,.3,14),(3.2,.3,21),(3.3,.5,24),(3.2,.5,24),(2.8,.7,24),(3.3,1,24)]:
        exact=mixture(lam,f,3.57129,8.01518,c,.041742)
        approx=erlang(np.array([lam]),np.array([3.57129*(1-f)+8.01518*f]),np.array([c]),.041742)
        print(lam,f,c,'exact',exact,'approx',[a.item() for a in approx],flush=True)
