import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,numpy as np
sys.path.insert(0,'/app/repo');import train as m
D=np.load('/app/data/sample.npz');V=np.load('/app/data/dev.npz');cfg=json.load(open('/app/repo/config.json'))
for lr in [.0015,.002,.003,.004,.005]:
 for noise in [.1,.2,.3,.4,.5]:
  c=dict(cfg,lr=lr,label_noise=noise,decay_powers=[.75])
  P,n=m.train(D['X'],D['y'],1000,c,0)
  print(lr,noise,np.mean(m.predict(P,n,V['X'])==V['y']),flush=True)
