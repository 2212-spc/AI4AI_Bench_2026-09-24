import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
S=np.load("/app/data/sample.npz"); D=np.load("/app/data/dev.npz")
Xs=S["X"].astype(np.float64); yd=D["y"]
MU=Xs.mean(0); L=np.linalg.cholesky(np.cov(Xs.T))
GOLD_PRIOR=np.bincount(yd,minlength=10)/len(yd)

def make_teacher(seed, h=64, temp=1.0, gain=1.0):
    r=np.random.default_rng(seed)
    d=32
    W1=gain*r.standard_normal((d,h))/np.sqrt(d); b1=r.standard_normal(h)*0.5
    W2=gain*r.standard_normal((h,h))/np.sqrt(h); b2=r.standard_normal(h)*0.5
    W3=r.standard_normal((h,10))/np.sqrt(h)
    # bias to get imbalanced class priors close to gold prior
    b3=np.log(GOLD_PRIOR+1e-3)*1.0
    return (W1,b1,W2,b2,W3,b3,temp)

def teacher_logits(T,Z):
    W1,b1,W2,b2,W3,b3,temp=T
    h=np.tanh(Z@W1+b1); h=np.tanh(h@W2+b2); return (h@W3+b3)*temp

def gen(T, n, seed, eps=0.3):
    r=np.random.default_rng(seed)
    Z=r.standard_normal((n,32)); X=Z@L.T+MU
    z=teacher_logits(T,Z); p=np.exp(z-z.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
    # sample gold label from teacher posterior (so Bayes rate < 1) 
    c=p.cumsum(1); u=r.random((n,1)); y=(u>c).sum(1).clip(0,9)
    flip=r.random(n)<eps; y_noisy=np.where(flip, r.integers(0,10,n), y)
    return X.astype(np.float32), y.astype(np.int64), y_noisy.astype(np.int64), p


if __name__=="__main__":
    import sys, json; sys.path.insert(0,"/app/repo"); import train as Tr
    cfg=json.load(open("/app/repo/config.json"))
    gain=float(sys.argv[1]); seed=int(sys.argv[2])
    for temp in [3.0,4.0,6.0]:
        T=make_teacher(seed,temp=temp,gain=gain)
        Xte,yte,_,pte=gen(T,4000,100)
        bayes=(pte.argmax(1)==yte).mean(); prior=np.bincount(yte,minlength=10)/4000
        res=[]
        for n,st in [(1000,250),(4000,1000)]:
            Xtr,_,ytr,_=gen(T,n,101)
            P,norm=Tr.train(Xtr,ytr,st,cfg,seed=0); res.append((Tr.predict(P,norm,Xte)==yte).mean())
        print(f"gain {gain} seed {seed} temp {temp}: bayes {bayes:.3f} prior_max {prior.max():.2f} mlp1000 {res[0]:.3f} mlp4000 {res[1]:.3f}", flush=True)
