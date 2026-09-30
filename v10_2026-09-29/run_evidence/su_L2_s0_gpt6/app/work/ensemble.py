import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,numpy as np,json
sys.path.insert(0,'/app/work');import original_train as tr
a=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');b=json.load(open('/app/work/original_config.json'))
for wd in [1,2,3]:
 for h in [64,128,256]:
  for q in [2,4,8]:
   logits=[]
   for seed in range(q):
    c=dict(b,weight_decay=wd,hidden=h);P,n=tr.train(a['X'],a['y'],1000,c,seed); zz=[]
    for i in range(0,len(d['X']),4096):zz.append(tr.forward(P,n(d['X'][i:i+4096]))[0])
    logits.append(np.vstack(zz))
   for typ,zz in [('mean',np.mean(logits,0)),('vote',None)]:
    if typ=='vote': pr=np.array([x.argmax(1) for x in logits]); out=np.array([np.bincount(pr[:,i],minlength=10).argmax() for i in range(len(pr[0]))])
    else:out=zz.argmax(1)
    print(wd,h,q,typ,np.mean(out==d['y']),flush=True)
