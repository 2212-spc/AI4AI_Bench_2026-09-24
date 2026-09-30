import json, numpy as np, sys
from itertools import product
from nm import multi
rows=[json.loads(l) for l in open('/app/lab_log.jsonl')]
onset=lambda n: 0.0157*(n/256)**-1.39
D=[]
for r in rows:
    a=r['args']; res=r['result']
    if res['diverged'] or res['final_loss'] is None or res['final_loss']>7: continue
    n,S,eta=a['n'],a['steps'],a['eta']
    if eta>onset(n)*1.0: continue
    D.append((n,S,eta,res['final_loss']))
D=np.array(D); n,S,eta,L=D.T
models={
 'nS_etaS': (lambda p,n,S,eta: p[0]+p[1]*n**-p[2]+p[3]*S**-p[4]+p[5]*(eta*S)**-p[6], [3,10,.5,10,.5,1,.5]),
 'n_etaS':  (lambda p,n,S,eta: p[0]+p[1]*n**-p[2]+p[3]*(eta*S)**-p[4], [3,10,.5,1,.5]),
 'n_S_eta': (lambda p,n,S,eta: p[0]+p[1]*n**-p[2]+p[3]*S**-p[4]*eta**-p[5], [3,10,.5,1,.5,.3]),
}
name=sys.argv[1] if len(sys.argv)>1 else 'nS_etaS'
m,p0=models[name]
f=lambda p: np.sum((m(p,n,S,eta)-L)**2) if np.all(np.isfinite(m(p,n,S,eta))) else 1e9
x0s=[]
for E0 in [2.0,3.0]:
    for sc in [0.5,1.0,2.0]:
        x0s.append([E0]+[v*sc if i%2==0 else v for i,v in enumerate(p0[1:])])
p,v=multi(f,x0s,restarts=4,maxiter=20000)
print(name, np.round(p,4), 'rmse',np.sqrt(v/len(L)))
res=m(p,n,S,eta)-L
for row,rr in zip(D,res): print(row, round(rr,4))
np.save(f'p_{name}.npy',p)
