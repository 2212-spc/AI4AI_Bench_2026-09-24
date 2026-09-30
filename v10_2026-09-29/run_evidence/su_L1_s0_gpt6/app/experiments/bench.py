import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,time,json,numpy as np
sys.path.insert(0,'/app/repo')
import train as m
D=np.load('/app/data/sample.npz');V=np.load('/app/data/dev.npz'); X,y=D['X'],D['y'];vx,vy=V['X'],V['y']
cfg=json.load(open('/app/repo/config.json'))
for steps,wd in [(1000,2.),(20000,2.),(1000,.1),(20000,.1),(20000,.5)]:
 c=dict(cfg,weight_decay=wd);t=time.time(); P,norm=m.train(X,y,steps,c,seed=0)
 print(steps,wd,'dev',np.mean(m.predict(P,norm,vx)==vy),'train',np.mean(m.predict(P,norm,X)==y),'secs',time.time()-t,flush=True)
