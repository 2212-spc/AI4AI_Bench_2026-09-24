import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,numpy as np,json,time
sys.path.insert(0,'/app/work');import original_train as tr
a=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz'); X=np.tile(a['X'],(20,1)); y=np.tile(a['y'],20); base=json.load(open('/app/work/original_config.json'))
cs=[(256,2,0,128),(256,2,.02,128),(128,2,0,128),(128,1.5,0,128),(256,1.5,0,128),(64,1,0,128),(256,4,0,128),(256,2,0,256)]
for h,wd,sm,bs in cs:
 vals=[];t0=time.time()
 for seed in [0,1]:
  c=dict(base,hidden=h,weight_decay=wd,label_smoothing=sm,batch_size=bs);P,n=tr.train(X,y,20000,c,seed); vals.append(float(np.mean(tr.predict(P,n,d['X'])==d['y'])))
 print((h,wd,sm,bs),vals,np.mean(vals),'wall',time.time()-t0,flush=True)
