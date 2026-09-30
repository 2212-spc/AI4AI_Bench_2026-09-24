import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json
sys.path.insert(0,'/app/work');import original_train as tr
A=np.load('/app/data/sample.npz');D=np.load('/app/data/dev.npz');X,y=A['X'],A['y'];V,z=D['X'],D['y'];
print('corr',np.round(np.corrcoef(np.column_stack([X,y]).T)[-1,:-1],2))
# class means standardized distance
mu=X.mean(0);sd=X.std(0); XX=(X-mu)/sd; VV=(V-mu)/sd
for c in range(10): print(c,'crowd',np.mean(y==c),'gold',np.mean(z==c),'means',np.round(XX[y==c].mean(0),1))
# MLP confusion
cfg=json.load(open('/app/work/original_config.json')); P,n=tr.train(X,y,1000,cfg,0); pred=tr.predict(P,n,V)
print('pred dist',np.bincount(pred,minlength=10)/len(pred));print(np.array([[np.sum((z==i)&(pred==j)) for j in range(10)] for i in range(10)]))
# feature marginal correlations to gold dev
print('goldcorr',np.round([np.corrcoef(V[:,j],z)[0,1] for j in range(32)],2))
