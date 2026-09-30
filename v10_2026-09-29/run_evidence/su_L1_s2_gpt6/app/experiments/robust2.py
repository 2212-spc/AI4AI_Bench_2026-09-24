import sys,json,itertools
sys.path.insert(0,'/app/experiments');import custom as train
import numpy as np
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');base=json.load(open('/app/repo/config.json'))
for mode,wd in [('gce.5',1),('gce.5',2),('focal1',1),('focal2',1),('boot.7',1),('boot.9',1),('sce.01',1),('sce.1',1)]:
 train.softmax_xent_grad.mode=mode;aa=[]
 for seed in [0,1]:
  p,n=train.train(s['X'],s['y'],1000,dict(base,weight_decay=wd),seed)
  aa.append(float(np.mean(train.predict(p,n,d['X'])==d['y'])))
 print(mode,wd,aa,np.mean(aa),flush=True)
