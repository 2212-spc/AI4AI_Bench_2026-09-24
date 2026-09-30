import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,numpy as np,json,time
sys.path.insert(0,'/app/work');import original_train as tr
a=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X=np.tile(a['X'],(20,1));y=np.tile(a['y'],20);b=json.load(open('/app/work/original_config.json')); h,wd,sm,bs=map(float,sys.argv[1:]);c=dict(b,hidden=int(h),weight_decay=wd,label_smoothing=sm,batch_size=int(bs));t=time.time();P,n=tr.train(X,y,20000,c,0);print('result',h,wd,sm,bs,np.mean(tr.predict(P,n,d['X'])==d['y']),time.time()-t,flush=True)
