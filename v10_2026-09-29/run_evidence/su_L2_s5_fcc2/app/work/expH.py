from exp_lib import *
rng=np.random.default_rng(31)
for n in [1800,3688]:
    subs=[rng.choice(len(Xu),n,replace=False) if n<len(Xu) else np.arange(len(Xu)) for _ in range(3)]
    norms=[tn.standardizer(Xu[idx]) for idx in subs]
    def ev(kind,par,wd):
        accs=[]
        for s in range(3):
            idx=subs[s]; Xs=norms[s](Xu[idx]); Xds=norms[s](Xd)
            P=train_loss(Xs,yu[idx],int(32*n/128),wd,kind,par,seed=s); accs.append((tn.predict_proba(P,Xds).argmax(1)==yd).mean())
        return np.mean(accs)
    print('n=%d'%n)
    for kind,par in [('ce',0),('gce',0.5),('gce',0.65),('gce',0.8),('forward',0.3),('forward',0.4),('forward',0.5)]:
        print('  %-8s %.2f: '%(kind,par)+'  '.join('wd%g %.3f'%(wd,ev(kind,par,wd)) for wd in [0.5,1.0,1.5,2.0,3.0,5.0]),flush=True)
