import sys, json, numpy as np
import train2 as T
s=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz')
X,y=s['X'].astype(np.float32),s['y'].astype(np.int64); Xd,yd=d['X'].astype(np.float32),d['y'].astype(np.int64)
base=json.load(open('/app/repo/config.json'))
def run(n, seed=0, val=0, **kw):
    cfg=dict(base); cfg.update(kw)
    rng=np.random.default_rng(100+seed)
    perm=rng.permutation(len(X)); idx=perm[:n]; vidx=perm[n:n+val]
    steps=int(32*n/128)
    M,norm=T.train(X[idx],y[idx],steps,cfg,seed=seed)
    r=[(T.predict(M,norm,Xd)==yd).mean()]
    if val: r.append((T.predict(M,norm,X[vidx])==y[vidx]).mean())
    return r
if __name__=='__main__':
    n=int(sys.argv[1]); seeds=int(sys.argv[2])
    for arg in sys.argv[3:]:
        kw=json.loads(arg)
        accs=np.array([run(n,seed=s,**kw) for s in range(seeds)])
        print(n, kw, ' '.join('%.3f'%v for v in accs.mean(0)), '+- %.3f'%(accs[:,0].std()/np.sqrt(seeds)), flush=True)
