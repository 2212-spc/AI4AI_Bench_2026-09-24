import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,numpy as np,json
sys.path.insert(0,'/app/work');import original_train as tr
a=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');b=json.load(open('/app/work/original_config.json')); crowd=np.bincount(a['y'],minlength=10)/len(a['y'])
# train once, adjust logits; then train weighted manually variant
for wd in [1,2,3]:
 P,n=tr.train(a['X'],a['y'],1000,dict(b,weight_decay=wd),0)
 # get logits
 V=d['X']; out=[]
 for i in range(0,len(V),4096):out.append(tr.forward(P,n(V[i:i+4096]))[0])
 zz=np.vstack(out); print('wd',wd)
 for alpha in [-2,-1,-.5,0,.5,1,2]:
  # divide by crowd prior^alpha; alpha negative boosts rare
  pp=zz-alpha*np.log(crowd);print(alpha,np.mean(pp.argmax(1)==d['y']),np.bincount(pp.argmax(1),minlength=10))
# weighted source
p='/app/work/base2.py';s=open(p).read();old='''        loss, dz = softmax_xent_grad(z, yb, k, cfg["label_smoothing"])
        G = backward(P, cache, dz)''';new='''        loss, dz = softmax_xent_grad(z, yb, k, cfg["label_smoothing"])
        cw = cfg.get("class_weights")
        if cw is not None:
            ww = np.asarray(cw, dtype=np.float32)[yb]; loss = (loss * ww).mean() / ww.mean(); dz *= (ww / ww.mean())[:,None]
        G = backward(P, cache, dz)''';assert old in s;s=s.replace(old,new);open('/app/work/weighted.py','w').write(s)
