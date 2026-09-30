import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
def run(kind):
 if kind=='sq': F=lambda x:np.concatenate([x,x*x],1)
 if kind=='abs': F=lambda x:np.concatenate([x,np.abs(x)],1)
 if kind=='all': F=lambda x:np.concatenate([x,x*x,np.abs(x)],1)
 if kind=='cube': F=lambda x:np.concatenate([x,x*x,x*x*x],1)
 xx=F(X);dd=F(D)
 for w in [1,2,3,4,5]:
  cc=dict(c,weight_decay=w);P,n=b.train(xx,y,1000,cc,seed=0);print(kind,w,(b.predict(P,n,dd)==Y).mean(),flush=True)
for k in ['sq','abs','all','cube']:run(k)
