import sys,json,itertools
sys.path.insert(0,'/app/experiments');import custom as train
import numpy as np
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');base=json.load(open('/app/repo/config.json'))
for mode,wd in itertools.product(['gce.7','gce.5','focal1','focal2','boot.7','boot.9','sce.01','sce.1'],[1.,2.,3.]):
 train.softmax_xent_grad.mode=mode
 aa=[]
 for seed in [0,1]:
  p,n=train.train(s['X'],s['y'],1000,dict(base,weight_decay=wd),seed)
  aa.append(float(np.mean(train.predict(p,n,d['X'])==d['y'])))
 print(mode,wd,aa,np.mean(aa),flush=True)
