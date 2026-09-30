import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,numpy as np,time
sys.path.insert(0,'/app/repo');import train as m
D=np.load('/app/data/sample.npz');V=np.load('/app/data/dev.npz');base=json.load(open('/app/repo/config.json'))
variants=[]
for noise in [.0,.1,.2,.3,.4,.5]: variants.append(('noise'+str(noise),dict(label_noise=noise,decay_powers=[.75])))
for h in [128,384,512]: variants.append(('h'+str(h),dict(hidden=h,label_noise=.3,decay_powers=[.75])))
for lr in [.001,.002,.004,.006]: variants.append(('lr'+str(lr),dict(lr=lr,label_noise=.3,decay_powers=[.75])))
for bs in [64,256,512,1024]: variants.append(('bs'+str(bs),dict(batch_size=bs,label_noise=.3,decay_powers=[.75])))
for name,v in variants:
 c=dict(base,**v);t=time.time();P,n=m.train(D['X'],D['y'],1000,c,0);a=np.mean(m.predict(P,n,V['X'])==V['y']);print(name,a,round(time.time()-t,2),flush=True)
