import sys, json, numpy as np
sys.path.insert(0,'/app/repo')
import train as T
s=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz')
X=s['X'].astype(np.float32); y=s['y'].astype(np.int64); Xd=d['X'].astype(np.float32); yd=d['y'].astype(np.int64)
BASE=json.load(open('/app/repo/config.json'))
def probs(P,norm,A):
    z,_=T.forward(P,norm(A)); z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def run(cfg,Xtr,ytr,steps,seed=0):
    c=dict(BASE); c.update(cfg)
    P,norm=T.train(Xtr,ytr,steps,c,seed=seed)
    return probs(P,norm,Xd)
