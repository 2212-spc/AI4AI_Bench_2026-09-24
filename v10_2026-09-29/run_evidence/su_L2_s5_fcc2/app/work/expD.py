from common import *
import importlib.util
spec=importlib.util.spec_from_file_location('tn','/app/work/train_new.py'); tn=importlib.util.module_from_spec(spec); spec.loader.exec_module(tn)
cfg=json.load(open('/app/work/config_new.json')); cfg['num_classes']=10
Xu, first = np.unique(X, axis=0, return_index=True); yu=y[first]
norm=tn.standardizer(Xu); Xs=norm(Xu); Xds=norm(Xd)
rng=np.random.default_rng(11)
print('(D) optimum wd vs number of unique training rows (32 epochs, 3 seeds, dev gold acc)')
for n in [900,1800,3688]:
    row=[]
    for wd in [1.0,1.5,2.0,3.0,4.5,7.0]:
        accs=[]
        for s in range(3):
            idx=rng.choice(len(Xu),n,replace=False) if n<len(Xu) else np.arange(len(Xu))
            P=tn.train_one(Xs[idx],yu[idx],int(32*n/128),cfg,wd,seed=s); accs.append((tn.predict_proba(P,Xds).argmax(1)==yd).mean())
        row.append('wd%.1f:%.3f'%(wd,np.mean(accs)))
    print('  n=%d '%n+' '.join(row),flush=True)
