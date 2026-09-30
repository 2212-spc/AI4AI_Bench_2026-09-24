import sys,json,time
sys.path.insert(0,'/app/repo')
import train
import numpy as np
s=np.load('/app/experiments/synthetic.npz'); X,y,yn=[s[k] for k in ['X','y','yn']]
cfg=json.load(open('/app/repo/config.json'))
for wd in map(float,sys.argv[1:]):
 t=time.process_time();p,norm=train.train(X[:80000],yn[:80000],20000,dict(cfg,weight_decay=wd))
 z,_=train.forward(p,norm(X[80000:]));np.save('/app/experiments/synth_logits_'+str(wd)+'.npy',z)
 print(json.dumps(dict(wd=wd,acc=float(np.mean(z.argmax(1)==y[80000:])),cpu=time.process_time()-t)),flush=True)
