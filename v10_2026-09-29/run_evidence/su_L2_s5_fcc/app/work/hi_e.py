import numpy as np
from exp import *; import t2
for e in [0.5,0.6]:
    for wd in [1.0,3.0,5.0]:
        a=[]
        for s in range(3):
            c=dict(BASE); c['weight_decay']=wd; c['noise_rate']=e; P,norm,_=t2.train(X,y,1000,c,seed=s); a.append((T.predict(P,norm,Xd)==yd).mean())
        print('SAMPLE e %.1f wd %.1f  %.3f'%(e,wd,np.mean(a)),flush=True)
