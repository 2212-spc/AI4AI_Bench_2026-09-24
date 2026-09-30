import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time,numpy as np
sys.path.insert(0,'/app/experiments')
import original_train as tr
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');cfg=json.load(open('/app/repo/config.json'))
for h in [32,64,128,256]:
 for wd in [.1,.3,1.,3.]:
  cfg.update(hidden=h,weight_decay=wd)
  for steps in [1000,4000]:
   t=time.process_time();p,norm=tr.train(s['X'],s['y'],steps,cfg)
   print(json.dumps(dict(h=h,wd=wd,steps=steps,acc=(tr.predict(p,norm,d['X'])==d['y']).mean(),trainacc=(tr.predict(p,norm,s['X'])==s['y']).mean(),cpu=time.process_time()-t)),flush=True)
