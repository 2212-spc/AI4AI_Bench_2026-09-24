import os, sys, json
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
from synth import make_teacher, gen
sys.path.insert(0,"/app/repo"); import train as Tr
cfg=json.load(open("/app/repo/config.json"))
name=sys.argv[1]; spec={"A":dict(seed=3,temp=4.0,gain=1.0),"B":dict(seed=1,temp=6.0,gain=2.0)}[name]
n=int(sys.argv[2]); wds=[float(w) for w in sys.argv[3].split(",")]
T=make_teacher(**spec)
Xte,yte,_,pte=gen(T,10000,100)
Xtr,ytr_gold,ytr,_=gen(T,n,101)
steps=n*32//128
for wd in wds:
    c=dict(cfg); c["weight_decay"]=wd
    P,norm=Tr.train(Xtr,ytr,steps,c,seed=0)
    acc=(Tr.predict(P,norm,Xte)==yte).mean()
    tr_noisy=(Tr.predict(P,norm,Xtr[:10000])==ytr[:10000]).mean()
    print(f"proxy {name} n {n} wd {wd}: test {acc:.3f} train-noisy {tr_noisy:.3f} bayes {(pte.argmax(1)==yte).mean():.3f}", flush=True)
