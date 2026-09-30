import sys,json,itertools
sys.path.insert(0,'/app/repo');import train
import numpy as np
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');base=json.load(open('/app/repo/config.json'))
for lr,wd,sm,h in itertools.product([.0015,.002,.003,.004],[.3,.6,1.,2.,3.,4.],[0,.03,.08,.15],[128,256]):
 c=dict(base,lr=lr,weight_decay=wd,label_smoothing=sm,hidden=h)
 p,n=train.train(s['X'],s['y'],1000,c,seed=0)
 a=float(np.mean(train.predict(p,n,d['X'])==d['y']))
 print(json.dumps(dict(lr=lr,wd=wd,sm=sm,h=h,a=a)),flush=True)
