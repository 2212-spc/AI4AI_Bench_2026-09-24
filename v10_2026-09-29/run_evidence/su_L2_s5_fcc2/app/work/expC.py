from common import *
import importlib.util
spec=importlib.util.spec_from_file_location('tn','/app/work/train_new.py'); tn=importlib.util.module_from_spec(spec); spec.loader.exec_module(tn)
cfg=json.load(open('/app/work/config_new.json')); cfg['num_classes']=10
Xu, first = np.unique(X, axis=0, return_index=True); yu=y[first]
norm=tn.standardizer(Xu); Xs=norm(Xu); Xds=norm(Xd)
rng=np.random.default_rng(3); perm=rng.permutation(len(Xu)); va=perm[:700]; tr=perm[700:]
print('(C) ranking consistency: val crowd agreement (700 unique rows) at 1/4 vs full budget; gold dev acc; 2 seeds')
for wd in [0.5,1.0,2.0,3.0,5.0,8.0]:
    r={}
    for frac,steps in [('1/4',int(32*len(tr)/128/4)),('full',int(32*len(tr)/128))]:
        va_acc=[];dv=[]
        for s in range(2):
            P=tn.train_one(Xs[tr],yu[tr],steps,cfg,wd,seed=s)
            va_acc.append((tn.predict_proba(P,Xs[va]).argmax(1)==yu[va]).mean()); dv.append((tn.predict_proba(P,Xds).argmax(1)==yd).mean())
        r[frac]=(np.mean(va_acc),np.mean(dv))
    print('  wd %-3g  1/4: val %.3f dev %.3f   full: val %.3f dev %.3f'%(wd,*r['1/4'],*r['full']),flush=True)
