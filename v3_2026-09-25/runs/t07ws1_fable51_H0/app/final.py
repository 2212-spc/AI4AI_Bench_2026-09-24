import numpy as np
from math import pi,sin,cos
exec(open('fit.py').read().split("np.random.seed(0)")[0])
np.random.seed(1)
def lrc(f): return 0.1+0.45*(1+cos(pi*f))
def pred(p,N,D,f,s,mode='raw'):
    l=lrc(f) if s=='cosine' else (1.0 if f<=0.8 else (1-f)/0.2)
    return model(p,np.array([N]),np.array([D]),np.array([f]),np.array([l]),np.array([area(s,f)]),mode)[0]
def answers(p,mode='raw'):
    C=1.08e21; opts={'A':2.7e8,'B':5.3e8,'C':1.1e9,'D':2.1e9}
    q1={k:pred(p,n,C/(6*n),1.0,'cosine',mode) for k,n in opts.items()}
    q2=pred(p,3e9,6e10,1.0,'wsd',mode)
    q3=pred(p,3e9,6e10,0.4,'cosine',mode)-pred(p,3e9,0.4*6e10,1.0,'wsd',mode)
    fin=pred(p,3e9,6e10,1.0,'cosine',mode)
    gaps=[pred(p,3e9,6e10,f,'cosine',mode)-fin for f in np.linspace(0.5,0.9,41)]
    q5=pred(p,3e9,6e10,0.5,'cosine',mode)-pred(p,3e9,0.35*6e10,1.0,'wsd',mode)
    return q1,q2,q3,(min(gaps),max(gaps)),q5
for mode,npar in [('raw',6),('raw',7)]:
    p,v=fit(mode,npar); print(mode,npar,'rmse %.4f'%np.sqrt(v),np.round(p,4))
    q1,q2,q3,q4,q5=answers(p,mode)
    print(' q1',{k:round(x,4) for k,x in q1.items()},' q2 %.4f q3 %.4f q4 [%.4f,%.4f] q5(cos0.5-wsd0.35) %.4f'%(q2,q3,q4[0],q4[1],q5))
# bootstrap over runs (resample run groups)
p0,_=fit('raw',6)
keys=sorted(set(zip(N,D,S))); groups=[np.where((N==k[0])&(D==k[1])&np.array([s==k[2] for s in S]))[0] for k in keys]
res=[]
for b in range(30):
    idx=np.concatenate([groups[i] for i in np.random.randint(len(groups),size=len(groups))])
    x0=list(p0)
    f=lambda q: np.mean((model(q,N[idx],D[idx],F[idx],LR[idx],AR[idx],'raw')-L[idx])**2)
    q,_=nm(f,x0,iters=6000)
    a=answers(q); res.append([a[1],a[2],a[3][0],a[3][1],a[4],a[0]['A'],a[0]['B'],a[0]['C'],a[0]['D']])
res=np.array(res); print('bootstrap mean',np.round(res.mean(0),4)); print('bootstrap std ',np.round(res.std(0),4))
print('q1 winner counts',{k:int(np.sum(res[:,5:].argmin(1)==i)) for i,k in enumerate('ABCD')})
