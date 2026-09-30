import sys,json,time
sys.path.insert(0,'/app/repo')
import train
import numpy as np
rng=np.random.default_rng(725)
X=rng.standard_normal((100000,32)).astype('float32')
# Independent synthetic nonlinear teacher, used only for a scale/runtime check.
p=train.init_params(32,128,10,rng)
z,_=train.forward(p,X)
# Match only aggregate class balance, not any real training examples.
prior=np.array([.189,.126,.004,.009,.082,.045,.012,.004,.192,.337])
bias=np.zeros(10)
for it in range(100):
 f=np.bincount((z+bias).argmax(1),minlength=10)/len(z)
 bias+=.5*(prior-f)
y=(z+bias).argmax(1); yn=y.copy(); flip=rng.random(len(y))<.3;yn[flip]=rng.integers(0,10,flip.sum())
np.savez('/app/experiments/synthetic.npz',X=X,y=y,yn=yn)
cfg=json.load(open('/app/repo/config.json'))
for n in [4000]:
 for wd in [1.,2.,3.,5.]:
  p,norm=train.train(X[:n],yn[:n],n//4,dict(cfg,weight_decay=wd))
  print(n,wd,np.mean(train.predict(p,norm,X[80000:])==y[80000:]),flush=True)
