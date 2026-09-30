from exp_lib import *
rng=np.random.default_rng(22)
for n in [900,1800,3688]:
    r={'mlp':[],'lda':[],'mlp+lda':[],'mlp3seeds':[],'mlp3+lda':[]}
    for s in range(3):
        idx=rng.choice(len(Xu),n,replace=False) if n<len(Xu) else np.arange(len(Xu))
        norm=tn.standardizer(Xu[idx]); Xs=norm(Xu[idx]); Xds=norm(Xd)
        wd={900:5.0,1800:3.0,3688:2.5}[n]
        pm=[tn.predict_proba(train_loss(Xs,yu[idx],int(32*n/128),wd,seed=10*s+j),Xds) for j in range(3)]
        pl=lda_fit(Xs,yu[idx])(Xds)
        r['mlp'].append((pm[0].argmax(1)==yd).mean()); r['lda'].append((pl.argmax(1)==yd).mean())
        r['mlp+lda'].append((((pm[0]+pl)/2).argmax(1)==yd).mean()); r['mlp3seeds'].append((np.mean(pm,0).argmax(1)==yd).mean())
        r['mlp3+lda'].append(((np.mean(pm,0)+pl).argmax(1)==yd).mean())
    print('n=%d  '%n+'  '.join('%s %.3f'%(k,np.mean(v)) for k,v in r.items()),flush=True)
