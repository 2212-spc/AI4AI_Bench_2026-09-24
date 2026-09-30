import numpy as np, sys, json, time
sys.path.insert(0,'/app/repo'); import train as T
S=np.load('/app/work/synth.npz'); X,yc,yg,Xt,yt=S['X'],S['yc'],S['yg'],S['Xt'],S['yt']
BASE=json.load(open('/app/repo/config.json'))
n=int(sys.argv[1]); over=json.loads(sys.argv[2]); wds=[float(v) for v in sys.argv[3].split(',')]; seeds=[int(s) for s in sys.argv[4].split(',')]
for wd in wds:
  for sd in seeds:
    c=dict(BASE,**over,weight_decay=wd); t0=time.time()
    P,norm=T.train(X[:n],yc[:n],int(32*n/c['batch_size']),c,seed=sd)
    print(n,over,wd,'seed',sd,'test %.4f'%(T.predict(P,norm,Xt)==yt).mean(),'%.0fs'%(time.time()-t0),flush=True)
