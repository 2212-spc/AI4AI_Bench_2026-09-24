from exp_lib import *
rng=np.random.default_rng(21)
for n in [1800,3688]:
    subs=[rng.choice(len(Xu),n,replace=False) if n<len(Xu) else np.arange(len(Xu)) for _ in range(3)]
    def ev(kind,par,wd):
        accs=[]
        for s in range(3):
            idx=subs[s]; norm=tn.standardizer(Xu[idx]); Xs=norm(Xu[idx]); Xds=norm(Xd)
            P=train_loss(Xs,yu[idx],int(32*n/128),wd,kind,par,seed=s); accs.append((tn.predict_proba(P,Xds).argmax(1)==yd).mean())
        return np.mean(accs)
    print('n=%d'%n)
    for wd in [1.0,2.0,3.0]:
        line='  wd %g: ce %.3f'%(wd,ev('ce',0,wd))
        for e in [0.2,0.3,0.4]: line+='  fwd%.1f %.3f'%(e,ev('forward',e,wd))
        for q in [0.5,0.7]: line+='  gce%.1f %.3f'%(q,ev('gce',q,wd))
        line+='  sce0.1 %.3f'%ev('sce',0.1,wd)
        print(line,flush=True)
