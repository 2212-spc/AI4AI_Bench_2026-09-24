import sys, json, numpy as np, time; sys.path.insert(0,'/app/work'); import train_new as TN; from synth import make
cfg=json.load(open(sys.argv[1])); seed=int(sys.argv[2]); sep=float(sys.argv[3]) if len(sys.argv)>3 else 0.38
X,y,yg=make(80000,1,sep=sep); Xd,_,yd=make(10000,999,sep=sep)
t0=time.process_time(); models,norm=TN.train(X,y,20000,cfg,seed=seed)
print('E2E sep %.2f seed %d gold-test %.4f  cpu %.1fs'%(sep,seed,(TN.predict(models,norm,Xd)==yd).mean(),time.process_time()-t0),flush=True)
