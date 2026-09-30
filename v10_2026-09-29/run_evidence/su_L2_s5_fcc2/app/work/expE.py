from common import *
import importlib.util
spec=importlib.util.spec_from_file_location('tn','/app/work/train_new.py'); tn=importlib.util.module_from_spec(spec); spec.loader.exec_module(tn)
import train as T
cfg=json.load(open('/app/work/config_new.json')); base=json.load(open('/app/repo/config.json'))
rng=np.random.default_rng(5)
Xu, first = np.unique(X, axis=0, return_index=True)
for n in [1800,3688]:
    res={'pipeline':[], 'pipeline-no-final':[], 'best-single':[], 'baseline wd3':[]}
    for seed in range(4):
        idx=rng.choice(len(Xu),n,replace=False) if n<len(Xu) else np.arange(len(Xu))
        # build a duplicated 'corpus' from these unique rows so dedupe is exercised
        rows=np.concatenate([first[idx], first[rng.choice(idx, n//12)]])
        steps=int(32*len(rows)/128)
        Ps,norm,info=tn.train(X[rows],y[rows],steps,cfg,seed=seed,verbose=False)
        res['pipeline'].append((tn.predict(Ps,norm,Xd)==yd).mean())
        res['pipeline-no-final'].append((tn.predict(Ps[:-1],norm,Xd)==yd).mean())
        res['best-single'].append((tn.predict([info['models'][0]['P']],norm,Xd)==yd).mean())
        P,nb=T.train(X[rows],y[rows],steps,base,seed=seed); res['baseline wd3'].append((T.predict(P,nb,Xd)==yd).mean())
        print('  n=%d seed=%d kept=%s best_wd=%g  '%(n,seed,info['kept'],info['best_wd'])+'  '.join('%s %.3f'%(k,v[-1]) for k,v in res.items()),flush=True)
    print('n=%d MEAN: '%n+'  '.join('%s %.3f'%(k,np.mean(v)) for k,v in res.items()),flush=True)
