# selection-signal test: train on 2000 rows, eval on other 2000 noisy rows and on dev gold
import numpy as np, json, sys
from exp import *
rng=np.random.default_rng(7); perm=rng.permutation(len(X)); tr,va=perm[:2000],perm[2000:]
out=[]
for wd in [0.3,1.0,2.0,3.0,5.0,8.0]:
  for seed in range(3):
    c=dict(BASE); c['weight_decay']=wd
    P,norm=T.train(X[tr],y[tr],500,c,seed=seed)
    pv=T.predict(P,norm,X[va]); pd=T.predict(P,norm,Xd)
    out.append((wd,seed,(pv==y[va]).mean(),(pd==yd).mean()))
    print('wd %.1f seed %d  noisy-val %.3f  gold-dev %.3f'%out[-1], flush=True)
json.dump(out,open('sel.json','w'))
