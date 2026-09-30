"""Synthetic stand-in: 10 classes, gold priors like dev, class-conditional 2-component Gaussian mixtures in 32-d,
uniform label noise e=0.29. `sep` controls difficulty."""
import numpy as np
PRI = np.array([84,65,49,11,487,21,41,9,68,165],float); PRI/=PRI.sum()
def make(n, seed, sep=1.0, e=0.29, nonlin=True, d=32, k=10):
    g = np.random.default_rng(12345)          # fixed world
    mu = g.standard_normal((k,2,d))*sep
    if not nonlin: mu[:,1]=mu[:,0]
    L = g.standard_normal((d,d))*0.3+np.eye(d); cov_chol=np.linalg.cholesky(L@L.T)
    r = np.random.default_rng(seed)
    yg = r.choice(k, n, p=PRI); comp = r.integers(0,2,n)
    X = mu[yg,comp] + r.standard_normal((n,d))@cov_chol.T
    flip = r.random(n)<e; yc = np.where(flip, r.integers(0,k,n), yg)
    return X.astype(np.float32), yc.astype(np.int64), yg.astype(np.int64)
if __name__=='__main__':
    import sys; sys.path.insert(0,'/app/repo'); import train as T, json
    BASE=json.load(open('/app/repo/config.json'))
    Xd,_,yd = make(5000, 999)
    for sep in [0.6,0.8,1.0]:
        X,y,_=make(4000,1,sep=sep); Xd,_,yd=make(5000,999,sep=sep)
        for wd in [0.3,3.0]:
            c=dict(BASE); c['weight_decay']=wd; P,norm=T.train(X,y,1000,c,seed=0)
            print('sep',sep,'wd',wd,'gold acc',(T.predict(P,norm,Xd)==yd).mean(),flush=True)
