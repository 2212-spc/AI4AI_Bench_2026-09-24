import sys
sys.path.insert(0,'/app/repo')
import train as m
import numpy as np,json,time
D=np.load('/app/data/sample.npz');V=np.load('/app/data/dev.npz')
cfg=json.load(open('/app/repo/config.json'))
for seed in [0,1]:
 t=time.process_time();models,norm=m.train(D['X'],D['y'],1000,cfg,seed)
 print('seed',seed,'dev',np.mean(m.predict(models,norm,V['X'])==V['y']),'individual',[np.mean(m.predict([p],norm,V['X'])==V['y']) for p in models],'cpu',time.process_time()-t,flush=True)
