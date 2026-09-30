import numpy as np
from exp import *
def lda(Xtr,ytr,A,shrink=0.0):
    mu=Xtr.mean(0); sd=Xtr.std(0); Xs=(Xtr-mu)/sd; As=(A-mu)/sd
    means=np.stack([Xs[ytr==c].mean(0) if (ytr==c).any() else Xs.mean(0) for c in range(10)]); pri=np.bincount(ytr,minlength=10)/len(ytr)+1e-6
    cov=sum(((Xs[ytr==c]-means[c]).T@(Xs[ytr==c]-means[c])) for c in range(10))/len(ytr); cov=(1-shrink)*cov+shrink*np.eye(32)
    ic=np.linalg.inv(cov); return np.stack([-0.5*((As-means[c])@ic*(As-means[c])).sum(1)+np.log(pri[c]) for c in range(10)],1).argmax(1)
for n in [250,500,1000,2000,4000]:
    accs=[]
    for s in range(3):
        idx=np.random.default_rng(1000+s).choice(len(X),n,replace=False) if n<4000 else np.arange(4000)
        accs.append((lda(X[idx],y[idx],Xd)==yd).mean())
    print('LDA n=%d  %.3f'%(n,np.mean(accs)))
