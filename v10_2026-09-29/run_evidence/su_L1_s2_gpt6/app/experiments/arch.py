import sys,json,itertools
sys.path.insert(0,'/app/repo');import train
import numpy as np
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');base=json.load(open('/app/repo/config.json'))
for h,b in itertools.product([64,128,192,256,384,512],[64,128,256]):
 c=dict(base,hidden=h,batch_size=b,weight_decay=3)
 aa=[]
 for seed in [0,1]:
  p,n=train.train(s['X'],s['y'],1000,c,seed);pr=train.predict(p,n,d['X']);aa.append(np.mean(pr==d['y']))
 print(h,b,aa,np.mean(aa),flush=True)
