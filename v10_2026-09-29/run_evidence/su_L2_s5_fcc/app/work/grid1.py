import sys, numpy as np, itertools, json
from exp import *
wd_list=[0.0,0.1,0.3,1.0,2.0,3.0,5.0]
n_list=[250,500,1000,2000,4000]
part=int(sys.argv[1]); nparts=int(sys.argv[2])
jobs=list(itertools.product(n_list,wd_list,[0,1,2]))
res=[]
for i,(n,wd,seed) in enumerate(jobs):
    if i%nparts!=part: continue
    a,t=run(n=n,seed=seed,weight_decay=wd)
    res.append((n,wd,seed,a))
json.dump(res,open(f'grid1_{part}.json','w'))
