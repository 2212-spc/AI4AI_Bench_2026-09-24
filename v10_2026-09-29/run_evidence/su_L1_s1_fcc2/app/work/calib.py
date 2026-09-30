from proxy import *
import train as T, json, time
BASE=json.load(open('/app/repo/config.json'))
def acc(P,norm,A,t):
    z,_=T.forward(P,norm(A)); return (z.argmax(1)==t).mean()
rng=np.random.default_rng(0)
for kind,K,spread,shrink,within in [('nl',3,0.35,0.8,1.0),('nl',3,0.5,0.7,1.0),('nl',4,0.5,0.5,1.0),('nl',3,0.7,0.6,1.3),('nl',6,0.6,0.4,1.2)]:
    g=make_gen(kind,0.35,K=K,spread=spread,shrink=shrink,within=within)
    Xte,_,yte=g(20000,rng)
    Xb,_,yb=g(60000,rng)
    cfg=dict(BASE); cfg.update({'weight_decay':0.3})
    P,norm=T.train(Xb,yb,15000,cfg); ab=acc(P,norm,Xte,yte)
    Xs4,yc4,_=g(4000,rng); P,norm=T.train(Xs4,yc4,1000,BASE); a4=acc(P,norm,Xte,yte)
    print(f'{kind} K={K} spread={spread} shrink={shrink} within={within}: clean-60k acc {ab:.3f}; 4000-noisy baseline: {a4:.3f}',flush=True)
