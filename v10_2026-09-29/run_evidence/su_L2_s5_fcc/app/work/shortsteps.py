import numpy as np, sys
from exp import *
frac=float(sys.argv[1])
for wd in [0.3,1.0,2.0,3.0,5.0]:
    a=[run(seed=s,weight_decay=wd,steps=int(1000*frac))[0] for s in range(3)]
    print('frac %.2f wd %.1f  %.3f'%(frac,wd,np.mean(a)),flush=True)
