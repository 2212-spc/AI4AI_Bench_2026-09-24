import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];base=json.load(open('/app/repo/config.json'))
for w in [2,3,4,5]:
 for h in [128,256,512]:
  ps=[]
  for seed in range(5):
   P,n=b.train(X,y,1000,dict(base,weight_decay=w,hidden=h),seed);z=b.forward(P,n(D))[0];z-=z.max(1,keepdims=True);p=np.exp(z);ps.append(p/p.sum(1,keepdims=True))
  for m in [1,2,3,5]:print(w,h,m,round((np.mean(ps[:m],0).argmax(1)==Y).mean(),3),end=';')
  print(flush=True)
