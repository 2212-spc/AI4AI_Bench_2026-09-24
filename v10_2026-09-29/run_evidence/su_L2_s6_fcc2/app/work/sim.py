"""Synthetic corpus generator: noise-corrected class-conditional Gaussians fitted on the sample."""
import numpy as np
s=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz')
X,y=s['X'].astype(np.float64),s['y']; Xd,yd=d['X'],d['y']
K=10; EPS=0.18
def fit(eps=EPS):
    n=len(X); cnt=np.bincount(y,minlength=K)
    # crowd prior = (1-eps) gold prior + eps/K  -> gold prior
    pri=(cnt/n-eps/K)/(1-eps); pri=np.clip(pri,0,None); pri/=pri.sum()
    M1=X.mean(0); M2=X.T@X/n
    params=[]
    for c in range(K):
        Xc=X[y==c]; m1=Xc.mean(0); m2=Xc.T@Xc/len(Xc)
        w=eps/K/(cnt[c]/n)   # fraction of crowd-c rows that are noise
        m1c=(m1-w*M1)/(1-w); m2c=(m2-w*M2)/(1-w)
        C=m2c-np.outer(m1c,m1c)
        ev,V=np.linalg.eigh(C); ev=np.maximum(ev,0.02); C=(V*ev)@V.T
        params.append((m1c,C))
    return pri,params
def generate(n,seed,eps=EPS,pri=None,params=None):
    if pri is None: pri,params=fit(eps)
    r=np.random.default_rng(seed)
    yg=r.choice(K,n,p=pri); Xo=np.zeros((n,32))
    for c in range(K):
        m=yg==c
        if m.any(): Xo[m]=r.multivariate_normal(params[c][0],params[c][1],m.sum())
    flip=r.random(n)<eps; yc=np.where(flip,r.integers(0,K,n),yg)
    return Xo.astype(np.float32),yg,yc
if __name__=='__main__':
    pri,params=fit(); print('gold prior est',pri.round(3)); print('dev prior',(np.bincount(yd,minlength=K)/len(yd)).round(3))
    Xs,yg,yc=generate(100000,0)
    np.savez('/app/work/synth.npz',X=Xs[:80000],yg=yg[:80000],yc=yc[:80000],Xt=Xs[80000:],yt=yg[80000:])
    # Bayes-ish accuracy: QDA with true params
    import numpy.linalg as la
    Xt,yt=Xs[80000:],yg[80000:]
    sc=[]
    for c in range(K):
        m,C=params[c]; Ci=la.inv(C); D=Xt-m; sc.append(-0.5*np.einsum('ij,jk,ik->i',D,Ci,D)-0.5*la.slogdet(C)[1]+np.log(pri[c]+1e-12))
    print('synthetic Bayes acc',(np.array(sc).T.argmax(1)==yt).mean())
    # how does real-dev look under this model? (LDA/QDA fit on true params applied to real dev)
    sc=[]
    for c in range(K):
        m,C=params[c]; Ci=la.inv(C); D=Xd-m; sc.append(-0.5*np.einsum('ij,jk,ik->i',D,Ci,D)-0.5*la.slogdet(C)[1]+np.log(pri[c]+1e-12))
    print('noise-corrected QDA on real dev',(np.array(sc).T.argmax(1)==yd).mean())
