from proxy import *
import train as T, json
BASE=json.load(open('/app/repo/config.json'))
def acc(P,norm,A,t):
    z,_=T.forward(P,norm(A)); return (z.argmax(1)==t).mean()
rng=np.random.default_rng(1)
g=make_gen('lin',0.35)
Xte,_,yte=g(20000,rng)
for n in [4000,16000,80000]:
    Xn,yc,_=g(n,rng); steps=int(32*n/128)
    for wd in [0.1,0.3,1.0,3.0]:
        cfg=dict(BASE); cfg['weight_decay']=wd
        P,norm=T.train(Xn,yc,steps,cfg); print(f'lin n={n} wd={wd} acc={acc(P,norm,Xte,yte):.3f}',flush=True)
