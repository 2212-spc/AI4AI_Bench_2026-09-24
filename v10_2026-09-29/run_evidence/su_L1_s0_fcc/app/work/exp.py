import sys, json, numpy as np, time
sys.path.insert(0, '/app/repo')
import train as T
s=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz')
X,y=s['X'].astype(np.float32),s['y'].astype(np.int64); Xd,yd=d['X'].astype(np.float32),d['y'].astype(np.int64)
base=json.load(open('/app/repo/config.json'))
def run(n, seed=0, **kw):
    cfg=dict(base); cfg.update(kw)
    rng=np.random.default_rng(100+seed)
    idx=rng.choice(len(X), n, replace=False) if n<len(X) else np.arange(len(X))
    steps=int(32*n/128)
    P,norm=T.train(X[idx],y[idx],steps,cfg,seed=seed)
    return (T.predict(P,norm,Xd)==yd).mean(), (T.predict(P,norm,X[idx])==y[idx]).mean()
if __name__=='__main__':
    for arg in sys.argv[1:]:
        kw=json.loads(arg)
        n=kw.pop('n',4000)
        accs=[run(n,seed=s,**kw) for s in range(2)]
        print(n, kw, 'dev %.3f  train(crowd) %.3f'%tuple(np.mean(accs,0)), flush=True)
