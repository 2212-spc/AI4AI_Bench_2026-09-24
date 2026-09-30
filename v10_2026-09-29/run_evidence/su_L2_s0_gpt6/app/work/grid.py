import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,numpy as np,json,time
sys.path.insert(0,'/app/work');import original_train as tr
a=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');base=json.load(open('/app/work/original_config.json'))
configs=[]
for h in [32,64,128,256]:
 for wd in [.5,1,2,3,4,6,8]: configs.append(dict(base,hidden=h,weight_decay=wd))
for smooth in [.02,.05,.1,.2]: configs.append(dict(base,weight_decay=2,label_smoothing=smooth))
for bs in [64,256,512]: configs.append(dict(base,weight_decay=2,batch_size=bs))
for c in configs:
 vals=[]
 for seed in [0,1]:
  P,n=tr.train(a['X'],a['y'],1000,c,seed); vals.append(np.mean(tr.predict(P,n,d['X'])==d['y']))
 print(c['hidden'],c['weight_decay'],c['label_smoothing'],c['batch_size'],vals,np.mean(vals),flush=True)
