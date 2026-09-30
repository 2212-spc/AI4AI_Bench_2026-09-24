from common import *
import importlib.util
spec=importlib.util.spec_from_file_location('tn','/app/work/train_new.py'); tn=importlib.util.module_from_spec(spec); spec.loader.exec_module(tn)
cfg=json.load(open('/app/work/config_new.json')); cfg['num_classes']=10
Xu, first = np.unique(X, axis=0, return_index=True); yu=y[first]
norm=tn.standardizer(Xu); Xs_all=norm(X); Xs_u=norm(Xu); Xds=norm(Xd)
print('(A) dedupe vs keep duplicates, wd=3, 1000 steps, 4 seeds')
for name,(A,t) in {'with dups (4000)':(Xs_all,y),'unique (3688)':(Xs_u,yu)}.items():
    accs=[(tn.predict_proba(tn.train_one(A,t,1000,cfg,3.0,seed=s),Xds).argmax(1)==yd).mean() for s in range(4)]
    print('  %-18s dev %.3f (+-%.3f)'%(name,np.mean(accs),np.std(accs)/2),flush=True)
print('(B) ensembles vs epochs on unique rows, wd=3 (single-model budget = 1000 steps)')
for members,steps in [(1,1000),(1,500),(1,250),(3,333),(3,1000),(5,200),(5,1000),(2,500),(4,250)]:
    res=[]
    for rep in range(2):
        ps=[tn.predict_proba(tn.train_one(Xs_u,yu,steps,cfg,3.0,seed=10*rep+m),Xds) for m in range(members)]
        res.append((np.mean(ps,0).argmax(1)==yd).mean())
    print('  %d members x %4d steps (total %4d): dev %.3f'%(members,steps,members*steps,np.mean(res)),flush=True)
