import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import sys, numpy as np, json, time
sys.path.insert(0,'/app/work'); import original_train as tr
a=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz'); X,y=a['X'],a['y']; cfg=json.load(open('/app/work/original_config.json')); perm=np.random.default_rng(321).permutation(len(y)); results=[]
for n in [500,1000,2000,4000]:
 idx=perm[:n] if n<4000 else np.arange(n)
 for wd in [0.125,0.25,0.5,1,2,4,8]:
  for seed in [0,1]:
   c=dict(cfg,weight_decay=wd); t=time.process_time(); P,norm=tr.train(X[idx],y[idx],max(1,n//4),c,seed=seed)
   pred=tr.predict(P,norm,d['X']); acc=float(np.mean(pred==d['y'])); train=float(np.mean(tr.predict(P,norm,X[idx])==y[idx])); row=dict(n=n,wd=wd,seed=seed,acc=acc,train=train,cpu=time.process_time()-t); results.append(row); print(json.dumps(row),flush=True)
json.dump(results,open('/app/work/scaling.json','w'),indent=2)
print('MEANS')
for n in [500,1000,2000,4000]:
 print(n,[(wd,round(np.mean([r['acc'] for r in results if r['n']==n and r['wd']==wd]),4)) for wd in [0.125,0.25,0.5,1,2,4,8]])
