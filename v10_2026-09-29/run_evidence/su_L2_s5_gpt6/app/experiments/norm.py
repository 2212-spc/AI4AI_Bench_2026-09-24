import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/experiments'); import baseline as b
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];c=json.load(open('/app/repo/config.json'))
def run(kind):
 if kind=='std':mu=X.mean(0); sd=X.std(0)+1e-6;f=lambda a:(a-mu)/sd
 if kind=='clip':mu=X.mean(0); sd=X.std(0)+1e-6;f=lambda a:np.clip((a-mu)/sd,-3,3)
 if kind=='median':mu=np.median(X,0);sd=np.median(np.abs(X-mu),0)*1.4826+1e-6;f=lambda a:(a-mu)/sd
 if kind=='quant':
  lo=np.quantile(X,.01,0);hi=np.quantile(X,.99,0);mu=(lo+hi)/2;sd=(hi-lo)/2;f=lambda a:np.clip((a-mu)/sd,-3,3)
 # monkey train manually replace standardizer
 old=b.standardizer;b.standardizer=lambda a:(lambda q:f(q).astype('float32'))
 for wd in [2,3,4]:
  P,n=b.train(X,y,1000,dict(c,weight_decay=wd),seed=0);print(kind,wd,(b.predict(P,n,D)==Y).mean())
 b.standardizer=old
for k in ['std','clip','median','quant']:run(k)
