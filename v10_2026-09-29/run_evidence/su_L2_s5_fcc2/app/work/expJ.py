from common import *
import importlib.util
spec=importlib.util.spec_from_file_location('v3','/app/work/train_v3.py'); v3=importlib.util.module_from_spec(spec); spec.loader.exec_module(v3)
import train as T
cfg=json.load(open('/app/work/config_v3.json')); base=json.load(open('/app/repo/config.json'))
rng=np.random.default_rng(5)
Xu, first = np.unique(X, axis=0, return_index=True)
for n in [1800,3688]:
    res={'v3':[], 'v3 best-single':[], 'baseline':[]}
    for seed in range(4):
        idx=rng.choice(len(Xu),n,replace=False) if n<len(Xu) else np.arange(len(Xu))
        rows=np.concatenate([first[idx], first[rng.choice(idx, n//12)]]); steps=int(32*len(rows)/128)
        Ms,norm,info=v3.train(X[rows],y[rows],steps,cfg,seed=seed,verbose=False)
        res['v3'].append((v3.predict(Ms,norm,Xd)==yd).mean())
        res['v3 best-single'].append((v3.predict([info['cands'][0]['model']],norm,Xd)==yd).mean())
        P,nb=T.train(X[rows],y[rows],steps,base,seed=seed); res['baseline'].append((T.predict(P,nb,Xd)==yd).mean())
        print('  n=%d seed=%d kept=%s  '%(n,seed,info['kept'])+'  '.join('%s %.3f'%(k,v[-1]) for k,v in res.items()),flush=True)
    print('n=%d MEAN: '%n+'  '.join('%s %.3f'%(k,np.mean(v)) for k,v in res.items()),flush=True)
