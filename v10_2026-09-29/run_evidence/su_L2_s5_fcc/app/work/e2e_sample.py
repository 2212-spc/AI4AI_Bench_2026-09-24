import sys, json, numpy as np, io, contextlib; sys.path.insert(0,'/app/work'); import train_new as TN
from exp import X,y,Xd,yd
cfg=json.load(open(sys.argv[1])); over=json.loads(sys.argv[2]) if len(sys.argv)>2 else {}; cfg.update(over)
accs=[]
for seed in range(5):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): models,norm=TN.train(X,y,1000,cfg,seed=seed)
    sel=[l for l in buf.getvalue().splitlines() if l.startswith('selected')][0]
    a=(TN.predict(models,norm,Xd)==yd).mean(); accs.append(a); print('seed',seed,'dev %.3f'%a,sel,flush=True)
print('MEAN %.4f'%np.mean(accs), over)
