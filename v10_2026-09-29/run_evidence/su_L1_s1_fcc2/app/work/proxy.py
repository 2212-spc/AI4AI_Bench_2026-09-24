import numpy as np, sys
sys.path.insert(0,'/app/repo')
d=np.load('/app/data/dev.npz'); Xd=d['X'].astype(np.float64); yd=d['y']
s=np.load('/app/data/sample.npz'); mu=s['X'].mean(0); sd=s['X'].std(0)
Xds=(Xd-mu)/sd
PRI=np.bincount(yd,minlength=10)/1000.0
PRI=np.maximum(PRI,0.01); PRI/=PRI.sum()
M=np.array([Xds[yd==c].mean(0) for c in range(10)])
R=Xds-M[yd]; S=R.T@R/1000
L=np.linalg.cholesky(S)

def make_gen(kind, noise, seed=123, K=3, spread=1.0, shrink=0.7, within=1.0, noise_kind='uniform'):
    """Return sampler(n, rng) -> (X, y_crowd, y_gold) for a proxy population."""
    rng=np.random.default_rng(seed)
    if kind=='lin':
        C=M[:,None,:]; K_=1; w=1.0
    else:
        K_=K; w=within
        C=shrink*M[:,None,:]+spread*(rng.standard_normal((10,K,32))@L.T)
    # class-dependent noise matrix option: crowd confuses towards a random 'attractor' class per class
    Tm=np.full((10,10),noise/10.0); np.fill_diagonal(Tm,1-noise+noise/10.0)
    if noise_kind=='classdep':
        for c in range(10):
            a=rng.integers(0,10)
            Tm[c]=(1-noise)*np.eye(10)[c]+noise*(0.5*np.eye(10)[a]+0.5/10)
    def sample(n, rng):
        yg=rng.choice(10,n,p=PRI); comp=rng.integers(0,K_,n)
        Xc=C[yg,comp]+w*rng.standard_normal((n,32))@L.T
        u=rng.random(n); cum=np.cumsum(Tm[yg],1); yc=(u[:,None]>cum).sum(1)
        return (Xc*sd+mu).astype(np.float32), yc.astype(np.int64), yg.astype(np.int64)
    return sample
