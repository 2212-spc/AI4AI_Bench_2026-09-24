import os, sys, json
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np, time
from synth import make_teacher, gen
import train2 as Tr
cfg=json.load(open("/app/repo/config.json"))
name=sys.argv[1]; spec={"A":dict(seed=3,temp=4.0,gain=1.0),"B":dict(seed=1,temp=6.0,gain=2.0)}[name]
n=int(sys.argv[2]); overrides=[json.loads(o) for o in sys.argv[3:]]
T=make_teacher(**spec)
Xte,yte,_,pte=gen(T,10000,100)
Xtr,ytr_gold,ytr,_=gen(T,n,101)
steps=n*32//128
for ov in overrides:
    c=dict(cfg); c.update(ov)
    t0=time.process_time()
    ens=int(c.get("ensemble",1)); Z=0
    for sd in range(ens):
        P,norm=Tr.train(Xtr,ytr,steps,c,seed=sd)
        z,_=Tr.forward(P,norm(Xte)); z=z-z.max(1,keepdims=True); p=np.exp(z); Z=Z+p/p.sum(1,keepdims=True)
        if sd==0: print(f"   single-seed0 test {(z.argmax(1)==yte).mean():.3f}", flush=True)
    acc=(Z.argmax(1)==yte).mean()
    print(f"proxy {name} n {n} {json.dumps(ov)}: test {acc:.3f}  cpu {time.process_time()-t0:.0f}s", flush=True)
