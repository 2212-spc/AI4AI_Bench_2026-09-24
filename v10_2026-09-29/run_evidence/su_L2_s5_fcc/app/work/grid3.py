import sys, json, numpy as np, itertools
from exp import *; import t2
part=int(sys.argv[1]); nparts=int(sys.argv[2])
jobs=list(itertools.product([0.0,0.2,0.3,0.4],[0.1,0.3,1.0,3.0],range(3)))
res=[]
for i,(e,wd,s) in enumerate(jobs):
    if i%nparts!=part: continue
    c=dict(BASE); c['weight_decay']=wd; c['noise_rate']=e
    P,norm,_=t2.train(X,y,1000,c,seed=s); res.append((e,wd,s,(T.predict(P,norm,Xd)==yd).mean()))
json.dump(res,open(f'grid3_{part}.json','w'))
