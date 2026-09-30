import numpy as np
s=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz')
X,y=s['X'],s['y']; Xd,yd=d['X'],d['y']
def lda(Xtr,ytr,Xte,shr=0.0):
    means=np.array([Xtr[ytr==c].mean(0) for c in range(10)]); S=sum(np.cov((Xtr[ytr==c]-means[c]).T)*(ytr==c).sum() for c in range(10))/len(Xtr)
    S=(1-shr)*S+shr*np.eye(32)*np.trace(S)/32
    Si=np.linalg.inv(S); pri=np.log(np.bincount(ytr,minlength=10)/len(ytr)+1e-9)
    return np.array([Xte@Si@means[c]-0.5*means[c]@Si@means[c]+pri[c] for c in range(10)]).T.argmax(1)
def qda(Xtr,ytr,Xte,shr=0.3):
    sc=[]; pri=np.log(np.bincount(ytr,minlength=10)/len(ytr)+1e-9)
    Sp=np.cov(Xtr.T)
    for c in range(10):
        Xc=Xtr[ytr==c]; m=Xc.mean(0); C=(1-shr)*np.cov(Xc.T)+shr*Sp; Ci=np.linalg.inv(C); D=Xte-m
        sc.append(-0.5*np.einsum('ij,jk,ik->i',D,Ci,D)-0.5*np.linalg.slogdet(C)[1]+pri[c])
    return np.array(sc).T.argmax(1)
for n in [500,1000,2000,4000]:
    for sd in range(3):
        r=np.random.default_rng(100+sd); idx=r.choice(len(X),n,replace=False) if n<len(X) else np.arange(len(X))
        print(n,sd,'lda',(lda(X[idx],y[idx],Xd)==yd).mean(),'qda.3',(qda(X[idx],y[idx],Xd,0.3)==yd).mean(),'qda.7',(qda(X[idx],y[idx],Xd,0.7)==yd).mean())
