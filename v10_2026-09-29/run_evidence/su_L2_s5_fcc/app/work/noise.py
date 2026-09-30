import numpy as np
from exp import *
rng=np.random.default_rng(7); perm=rng.permutation(len(X)); tr,va=perm[:3000],perm[3000:]
c=dict(BASE); P,norm=T.train(X[tr],y[tr],750,c,seed=0)
pv=T.predict(P,norm,X[va])
M=np.zeros((10,10),int)
for p,t in zip(pv,y[va]): M[p,t]+=1
np.set_printoptions(linewidth=200)
print('rows=pred, cols=crowd label\n',M)
print('agreement',(pv==y[va]).mean())
# for confident preds of class 4 (majority), distribution of crowd labels
print('dev pred acc',(T.predict(P,norm,Xd)==yd).mean())
