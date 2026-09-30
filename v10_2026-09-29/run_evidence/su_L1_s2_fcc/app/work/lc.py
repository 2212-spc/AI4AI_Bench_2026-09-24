from harness import *
import numpy as np, itertools
res={}
for n in [500,1000,2000,4000]:
    for wd in [0.1,0.3,1.0,2.0,3.0]:
        a=[run({"weight_decay":wd}, n=n, seed=s, sub_seed=s) for s in range(2)]
        res[(n,wd)]=np.mean(a); print(n, wd, "%.3f"%np.mean(a), flush=True)
