import os, sys, json, itertools
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
sys.path.insert(0, '/app/repo')
import importlib
S=np.load('/app/data/sample.npz'); D=np.load('/app/data/dev.npz')
X,y=S['X'].astype(np.float32),S['y'].astype(np.int64); Xd,yd=D['X'].astype(np.float32),D['y']
BASE=json.load(open('/app/repo/config.json'))

def run(cfg_over, n=4000, epochs=32, seeds=(0,1), module='train', sub_seed=0):
    m=importlib.import_module(module)
    cfg=dict(BASE); cfg.update(cfg_over)
    accs=[]
    for s in seeds:
        rng=np.random.default_rng(1000+sub_seed+s)
        idx=rng.choice(len(X), n, replace=False) if n<len(X) else np.arange(len(X))
        steps=int(epochs*n/cfg['batch_size'])
        P,norm=m.train(X[idx],y[idx],steps,cfg,seed=s,verbose=False) if module!="train_orig" else m.train(X[idx],y[idx],steps,cfg,seed=s)
        accs.append((m.predict(P,norm,Xd)==yd).mean())
    return float(np.mean(accs)), accs

if __name__=='__main__':
    # usage: harness.py module n epochs json_overrides...
    module=sys.argv[1]; n=int(sys.argv[2]); ep=float(sys.argv[3])
    for j in sys.argv[4:]:
        o=json.loads(j); mean,accs=run(o,n=n,epochs=ep,seeds=(0,1,2),module=module)
        print(f"{module} n={n} ep={ep} {j} -> {mean:.4f} {np.round(accs,3).tolist()}", flush=True)
