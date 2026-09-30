import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import sys, numpy as np, json,time
sys.path.insert(0,'/app/work'); import original_train as tr
a=np.load('/app/data/sample.npz'); d=np.load('/app/data/dev.npz'); X,y=a['X'],a['y']; cfg=json.load(open('/app/work/original_config.json')); perm=np.random.default_rng(321).permutation(len(y));
n=int(sys.argv[1]); wds=list(map(float,sys.argv[2:])); idx=perm[:n] if n<4000 else np.arange(n)
for wd in wds:
 for seed in [0,1]:
  c=dict(cfg,weight_decay=wd); t=time.process_time(); P,norm=tr.train(X[idx],y[idx],n//4,c,seed=seed)
  pred=tr.predict(P,norm,d['X']); row=dict(n=n,wd=wd,seed=seed,acc=float(np.mean(pred==d['y'])),train=float(np.mean(tr.predict(P,norm,X[idx])==y[idx])),cpu=time.process_time()-t)
  with open('/app/work/scaling_live.jsonl','a') as f: f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
