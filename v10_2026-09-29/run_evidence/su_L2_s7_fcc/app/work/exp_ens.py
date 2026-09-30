import numpy as np
from lib import *
import mlp
# ensembling gain: 3 seeds on 90% subsets vs single model on 100%
def softmax(z): z=z-z.max(1,keepdims=True); p=np.exp(z); return p/p.sum(1,keepdims=True)
for rep in range(2):
    single = run(dict(weight_decay=3.0, noise_rate=0.25), n=4000, seed=rep, mod=mlp)[0]
    probs=[]; accs=[]
    for s in range(3):
        a,P,norm = run(dict(weight_decay=3.0, noise_rate=0.25), n=3600, seed=10*rep+s, sub_seed=10*rep+s, mod=mlp)
        probs.append(softmax(mlp.logits(P,norm,Xd))); accs.append(a)
    ens = (np.mean(probs,0).argmax(1)==yd).mean()
    print('rep',rep,'single100%',single,'members',accs,'ens3',ens, flush=True)
