import sys, json, numpy as np, time; sys.path.insert(0,'/app/repo'); import train as T, t2; from synth import make
BASE=json.load(open('/app/repo/config.json'))
X,y,yg=make(80000,1,sep=0.38); Xd,_,yd=make(10000,999,sep=0.38)
for e,wd in [tuple(map(float,a.split(':'))) for a in sys.argv[1:]]:
    c=dict(BASE); c['weight_decay']=wd; c['noise_rate']=e; t0=time.time()
    P,norm,_=t2.train(X,y,20000,c,seed=0)
    print('SYN80k e %.1f wd %.2f  gold-test %.4f  %.0fs'%(e,wd,(T.predict(P,norm,Xd)==yd).mean(),time.time()-t0),flush=True)
