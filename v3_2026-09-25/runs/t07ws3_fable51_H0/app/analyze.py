import json,numpy as np
from itertools import product
runs=[json.loads(l) for l in open('/app/lab_runs.jsonl')]
# finished points: (N,D,loss,is_wsd)
pts=[(2e7,2.8e8,6.1069,0),(5e7,7e8,4.8526,0),(1.2e8,1.68e9,3.9984,0),(3e8,4.2e9,3.3555,0),(1.2e8,1.68e9,3.9949,1)]
ck={}
for r in runs:
    c=r['config'];N,D,s=c['N'],c['D'],c['sched']
    pts.append((N,D,r['loss'],int(s=='wsd')))
    ck[(N,D,s)]={k['frac']:k['loss'] for k in r['checkpoints']}
    if r.get('cooldown_branches'):
        ck[(N,D,'cd')]={k['frac']:k['loss'] for k in r['cooldown_branches']}
        for k in r['cooldown_branches']: pts.append((N,k['tokens'],k['loss'],1))
P=np.array(pts);N,D,L,W=P.T
def fit(mask,offset=True,grid=None):
    n,d,l,w=N[mask],D[mask],L[mask],W[mask]
    best=None
    for a,b in product(np.arange(0.25,0.75,0.005),np.arange(0.2,0.55,0.005)):
        cols=[np.ones_like(n),n**-a,d**-b]+([w] if offset else [])
        X=np.c_[tuple(cols)]
        coef=np.linalg.lstsq(X,l,rcond=None)[0]
        r=((X@coef-l)**2).sum()
        if best is None or r<best[0]: best=(r,a,b,coef)
    r,a,b,coef=best
    return dict(rms=np.sqrt(r/len(l)),a=a,b=b,E=coef[0],A=coef[1],B=coef[2],off=(coef[3] if offset else 0),n=len(l))
def pred(p,n,d,w=0): return p['E']+p['A']*n**-p['a']+p['B']*d**-p['b']+p['off']*w
C=1.08e21
opts=[8.9e8,1.8e9,3.6e9,7.1e9]
def report(name,p):
    print(f"{name}: n={p['n']} rms={p['rms']:.4f} a={p['a']:.3f} b={p['b']:.3f} E={p['E']:.3f} A={p['A']:.1f} B={p['B']:.1f} off={p['off']:.4f}")
    print('   q1 losses',[round(pred(p,n,C/(6*n)),4) for n in opts],' Nopt~',round(( (p['b']*p['B'])/(p['a']*p['A'])*(6/C)**-p['b'] )**(1/(p['a']+p['b'])) if False else 0),
          ' prod wsd',round(pred(p,3e9,6e10,1),4),' prod cos',round(pred(p,3e9,6e10,0),4))
    print('   D-term at prod',round(p['B']*6e10**-p['b'],4))
    return p
pj=report('joint all',fit(np.ones(len(L),bool)))
pc=report('cosine only',fit(W==0,offset=False))
pw=report('wsd only(all)',fit(W==1,offset=False))
# leave-one-out robustness for q1 ordering
import collections
cnt=collections.Counter()
for i in range(len(L)):
    m=np.ones(len(L),bool);m[i]=False
    p=fit(m); cnt[int(np.argmin([pred(p,n,C/(6*n)) for n in opts]))]+=1
print('LOO q1 argmin counts',cnt)
# residuals
for i in range(len(L)):
    print(f"  N={N[i]:.1e} D={D[i]:.2e} w={int(W[i])} L={L[i]:.4f} res={L[i]-pred(pj,N[i],D[i],W[i]):+.4f}")
json.dump(dict(pj=pj,pc=pc,pw=pw),open('/app/fits.json','w'),default=float)

print("\n=== cosine ckpt excess over wsd-law prediction at same tokens (joint fit, w=1) ===")
fr=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
ex={}
for (n,d,s),c in sorted(ck.items()):
    if s!='cosine': continue
    row=[c[f]-pred(pj,n,f*d,1) for f in fr]
    ex[(n,d)]=row
    print(f"N={n:.0e} D={d:.0e}", ' '.join(f"{v:+.3f}" for v in row))
E=np.array(list(ex.values())); print("mean ", ' '.join(f"{v:+.3f}" for v in E.mean(0))); print("std  ", ' '.join(f"{v:+.3f}" for v in E.std(0)))
print("\n=== cosine ckpt minus measured wsd branch at same tokens ===")
for (n,d,s),c in sorted(ck.items()):
    if s!='cd': continue
    cc=ck[(n,d,'cosine')]
    print(f"N={n:.0e} D={d:.0e}", {f:round(cc[f]-c[f],4) for f in c})
print("\n=== wsd stable ckpt minus branch ===")
for (n,d,s),c in sorted(ck.items()):
    if s!='cd': continue
    cw=ck[(n,d,'wsd')]
    print(f"N={n:.0e} D={d:.0e}", {f:round(cw[f]-c[f],4) for f in c})
print("\n=== q4 raw: cos ckpt(f) - cos final ===")
for (n,d,s),c in sorted(ck.items()):
    if s!='cosine': continue
    fin=[p for p in pts if p[0]==n and p[1]==d and p[3]==0][0][2]
    print(f"N={n:.0e} D={d:.0e}", {f:round(c[f]-fin,4) for f in [0.5,0.6,0.7,0.8,0.9]})
