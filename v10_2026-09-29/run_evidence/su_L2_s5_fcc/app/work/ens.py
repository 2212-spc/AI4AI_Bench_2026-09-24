import numpy as np
from exp import *
def probs(P,norm,A):
    z,_=T.forward(P,norm(A)); z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
for wd in [2.0,3.0]:
    ps=[]
    for s in range(5):
        c=dict(BASE); c['weight_decay']=wd; P,norm=T.train(X,y,1000,c,seed=s); ps.append(probs(P,norm,Xd))
        print('wd',wd,'seed',s,'single',(ps[-1].argmax(1)==yd).mean(),' ens%d'%len(ps),(np.mean(ps,0).argmax(1)==yd).mean(),flush=True)
