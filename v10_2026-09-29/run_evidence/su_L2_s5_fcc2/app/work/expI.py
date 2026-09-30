from exp_lib import *
rng=np.random.default_rng(32)
for n in [1800,3688]:
    r={}
    for s in range(3):
        idx=rng.choice(len(Xu),n,replace=False) if n<len(Xu) else np.arange(len(Xu))
        norm=tn.standardizer(Xu[idx]); Xs=norm(Xu[idx]); Xds=norm(Xd); steps=int(32*n/128)
        wd={1800:2.0,3688:1.5}[n]
        pg=[tn.predict_proba(train_loss(Xs,yu[idx],steps,wd,'gce',0.5,seed=10*s+j),Xds) for j in range(3)]
        pf=[tn.predict_proba(train_loss(Xs,yu[idx],steps,wd,'forward',0.4,seed=20*s+j),Xds) for j in range(2)]
        pl={sh:lda_fit(Xs,yu[idx],sh)(Xds) for sh in [0.0,0.02,0.1,0.3]}
        def acc(p): return (p.argmax(1)==yd).mean()
        cand={'gce1':acc(pg[0]),'gce3':acc(np.mean(pg,0)),'fwd1':acc(pf[0]),'gce1+fwd1':acc((pg[0]+pf[0])/2),
              'lda0':acc(pl[0.0]),'lda.02':acc(pl[0.02]),'lda.1':acc(pl[0.1]),'lda.3':acc(pl[0.3]),
              'gce1+lda':acc((pg[0]+pl[0.02])/2),'gce3+lda':acc((np.mean(pg,0)+pl[0.02])/2),'gce3+lda(w.5)':acc((np.mean(pg,0)+0.5*pl[0.02])),
              'gce3+fwd2+lda':acc((np.mean(pg,0)+np.mean(pf,0)+pl[0.02])),'gce1+lda(w.5)':acc(pg[0]+0.5*pl[0.02])}
        for k,v in cand.items(): r.setdefault(k,[]).append(v)
    print('n=%d  '%n+'  '.join('%s %.3f'%(k,np.mean(v)) for k,v in r.items()),flush=True)
