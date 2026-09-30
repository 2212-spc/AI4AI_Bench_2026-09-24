import sys, json, numpy as np, time; sys.path.insert(0,'/app/repo'); import train as T; from synth import make
BASE=json.load(open('/app/repo/config.json'))
sep=float(sys.argv[1]); wds=[float(w) for w in sys.argv[2].split(',')]; n=int(sys.argv[3]); steps=int(sys.argv[4])
X,y,yg=make(n,1,sep=sep); Xd,_,yd=make(10000,999,sep=sep)
for wd in wds:
    c=dict(BASE); c['weight_decay']=wd; t0=time.time(); P,norm=T.train(X,y,steps,c,seed=0)
    ptr=T.predict(P,norm,X[:10000])
    print('sep %.2f n %d steps %d wd %.2f  gold-test %.4f  train-crowd %.3f train-gold %.3f  %.0fs'%(sep,n,steps,wd,(T.predict(P,norm,Xd)==yd).mean(),(ptr==y[:10000]).mean(),(ptr==yg[:10000]).mean(),time.time()-t0),flush=True)
