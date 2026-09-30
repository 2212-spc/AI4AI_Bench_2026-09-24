import numpy as np
from proxy import make_gen
g=make_gen(kind='nl',K=4,spread=0.5,shrink=0.5,noise=0.35)
X,yc,yg=g(80000,np.random.default_rng(7)); np.savez('/app/work/corpusA.npz',X=X,y=yc)
Xt,_,yt=g(20000,np.random.default_rng(999)); np.save('/app/work/testA_X.npy',Xt); np.save('/app/work/testA_y.npy',yt)
