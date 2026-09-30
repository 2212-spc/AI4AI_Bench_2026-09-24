import json, numpy as np
from itertools import product
nb=json.load(open('notebook/checkpoints.json'))
runs=[json.loads(l) for l in open('lab_runs.jsonl')]
# finished points
cos=[(2e7,2.8e8,7.1749),(5e7,7e8,5.7031),(1.2e8,1.68e9,4.6744),(3e8,4.2e9,3.8907)]
wsd=[(1.2e8,1.68e9,4.6487)]
ck={}  # (N,D,sched)->list of (f,loss)
for r in runs:
    c=r['config']; key=(c['N'],c['D'],c['sched'])
    if r['status']!='ok': continue
    (cos if c['sched']=='cosine' else wsd).append((c['N'],c['D'],r['loss']))
    ck[key]=[(x['frac'],x['loss']) for x in r['checkpoints']]
    for x in r.get('cooldown_branches') or []:
        if x['frac']<1.0: wsd.append((c['N'],x['tokens'],x['loss']))
cos=np.array(cos); wsd=np.array(wsd)
np.save('cos.npy',cos); np.save('wsd.npy',wsd)
def fit(data, floor_grid=None):
    # L = E + A N^-a + B D^-b ; grid over a,b,E then linear for A,B
    N,D,L=data.T
    best=None
    for a in np.linspace(0.1,1.2,111):
        for b in np.linspace(0.1,1.2,111):
            X=np.c_[np.ones_like(N), N**-a, D**-b]
            coef,res,_,_=np.linalg.lstsq(X,L,rcond=None)
            r=L-X@coef; s=(r**2).sum()
            if coef[1]>0 and coef[2]>0 and (best is None or s<best[0]): best=(s,a,b,coef)
    return best
for name,data in [('cosine',cos),('wsd',wsd)]:
    s,a,b,(E,A,B)=fit(data)
    print(name,len(data),'rms',np.sqrt(s/len(data)),'alpha',a,'beta',b,'E',E,'A',A,'B',B)
    pred=lambda N,D: E+A*N**-a+B*D**-b
    print('  prod 3e9,6e10 ->',pred(3e9,6e10),' ; q1 options:',[(n,round(pred(n,1.08e21/(6*n)),4)) for n in [1.2e9,2.4e9,4.8e9,9.6e9]])
    for N,D,L in data: print('   ',N,D,L,round(pred(N,D)-L,4))
