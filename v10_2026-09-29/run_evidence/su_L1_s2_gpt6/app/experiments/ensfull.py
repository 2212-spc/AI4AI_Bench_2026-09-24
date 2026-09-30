import sys,json,time
sys.path.insert(0,'/app/repo');import train
import numpy as np
s=np.load('/app/experiments/synthetic.npz');X,y,yn=[s[k] for k in ['X','y','yn']];base=json.load(open('/app/repo/config.json'))
for spec in [[(.3,0),(.6,1),(1.,2) ],[(.3,0),(.3,1),(.6,2)],[(.6,0),(.6,1),(.6,2)]]:
 zs=[];t=time.process_time()
 for wd,seed in spec:
  p,n=train.train(X[:80000],yn[:80000],20000,dict(base,weight_decay=wd),seed)
  z,_=train.forward(p,n,X[80000:]);zs.append(z)
 print(spec,float(np.mean(np.mean(zs,axis=0).argmax(1)==y[80000:])),time.process_time()-t,flush=True)
