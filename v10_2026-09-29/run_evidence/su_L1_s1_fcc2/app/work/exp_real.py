import numpy as np, json, sys
sys.path.insert(0,'/app/work'); import train2 as T2
s=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz')
X=s['X'].astype(np.float32); y=s['y']; Xd=d['X'].astype(np.float32); yd=d['y']
BASE=json.load(open('/app/repo/config.json'))
def ev(cfg,seeds=(0,1,2)):
    c=dict(BASE); c.update(cfg); accs=[]
    for sd in seeds:
        m,norm=T2.train(X,y,1000,c,seed=sd); accs.append((T2.predict(m,norm,Xd)==yd).mean())
    return np.mean(accs), np.std(accs)
for cfg in [{}, {'avg_frac':0.25}, {'avg_frac':0.5}, {'ensemble':3}, {'ensemble':3,'avg_frac':0.25},
            {'hidden':512}, {'hidden':512,'weight_decay':2.0}, {'label_smoothing':0.1}, {'lr':0.001,'weight_decay':2.0}]:
    print(cfg, 'dev acc %.3f +- %.3f'%ev(cfg), flush=True)
