import numpy as np
from exp import *; import t2
rng=np.random.default_rng(7); perm=rng.permutation(len(X)); tr,va=perm[:3600],perm[3600:]
for wd,e in [(0.3,0.4),(3.0,0.4),(1.0,0.4)]:
    c=dict(BASE); c['weight_decay']=wd; c['noise_rate']=e
    print('wd',wd,'e',e)
    P,norm,h1=t2.train(X[tr],y[tr],900,c,seed=0,Xv=X[va],yv=y[va],eval_every=50)
    P,norm,h2=t2.train(X[tr],y[tr],900,c,seed=0,Xv=Xd,yv=yd,eval_every=50)
    for (t,a),(t2_,b) in zip(h1,h2): print('  step %4d noisy-val %.3f gold-dev %.3f'%(t,a,b),flush=True)
