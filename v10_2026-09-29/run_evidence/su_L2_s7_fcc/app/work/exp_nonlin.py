import numpy as np
from lib import *
mu, sd = X.mean(0), X.std(0)+1e-6
Xs, Xds = (X-mu)/sd, (Xd-mu)/sd
def logreg(Xt, yt, Xv, l2=1e-3, its=3000, lr=0.3):
    W = np.zeros((Xt.shape[1],10)); b=np.zeros(10)
    for it in range(its):
        z = Xt@W+b; z-=z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True)
        p[np.arange(len(yt)),yt]-=1; p/=len(yt)
        W -= lr*(Xt.T@p + l2*W); b-=lr*p.sum(0)
    return (Xv@W+b).argmax(1)
print('linear', (logreg(Xs,y,Xds)==yd).mean(), flush=True)
Q = lambda A: np.hstack([A, (A**2-1)/np.sqrt(2)])
for l2 in (1e-3,1e-2): print('lin+sq', l2, (logreg(Q(Xs),y,Q(Xds),l2)==yd).mean(), flush=True)
iu = np.triu_indices(32,1)
Q2 = lambda A: np.hstack([A, (A**2-1)/np.sqrt(2), (A[:,:,None]*A[:,None,:])[:,iu[0],iu[1]]])
for l2 in (1e-2,3e-2,1e-1): print('full quad', l2, (logreg(Q2(Xs),y,Q2(Xds),l2,its=3000,lr=0.1)==yd).mean(), flush=True)
for D in (500, 2000):
    r=np.random.default_rng(0); Wr=r.standard_normal((32,D)).astype(np.float32)/np.sqrt(32); br=r.standard_normal(D)*0.5
    R = lambda A: np.maximum(A@Wr+br,0)*np.sqrt(2.0/D)*4
    for l2 in (1e-3,1e-2,3e-2): print('rf', D, l2, (logreg(R(Xs),y,R(Xds),l2,its=3000,lr=0.3)==yd).mean(), flush=True)
