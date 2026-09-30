import json, numpy as np
runs=[json.loads(l) for l in open('lab_runs.jsonl')]
nb=json.load(open('notebook/checkpoints.json'))
nbcfg={1:(2e7,2.8e8,'cosine',7.1749),2:(5e7,7e8,'cosine',5.7031),3:(1.2e8,1.68e9,'cosine',4.6744),4:(3e8,4.2e9,'cosine',3.8907),5:(1.2e8,1.68e9,'wsd',4.6487)}
def lr_cos(f): return 0.1+0.9*0.5*(1+np.cos(np.pi*f))
def lr_wsd(f): return np.minimum(1.0,(1-f)/0.2)
pts=[]  # N, tokens, lr, loss, tag
def add_run(N,D,s,final,cks,cools):
    pts.append((N,D,0.1 if s=='cosine' else 0.0,final,'fin_'+s))
    for f,l in cks:
        pts.append((N,f*D,lr_cos(f) if s=='cosine' else lr_wsd(f),l,'ck_'+s))
    for f,l in cools:
        if f<1.0: pts.append((N,f*D,0.0,l,'cool'))
for e in nb:
    N,D,s,fin=nbcfg[e['run']]; add_run(N,D,s,fin,[(x['frac'],x['loss']) for x in e['checkpoints']],[])
for r in runs:
    c=r['config']; add_run(c['N'],c['D'],c['sched'],r['loss'],[(x['frac'],x['loss']) for x in r['checkpoints']],[(x['frac'],x['loss']) for x in r.get('cooldown_branches') or []])
N=np.array([p[0] for p in pts]); T=np.array([p[1] for p in pts]); LR=np.array([p[2] for p in pts]); L=np.array([p[3] for p in pts]); tag=np.array([p[4] for p in pts])
print(len(pts),'points; by tag',{t:(tag==t).sum() for t in set(tag)})
def solve(a,b,p,mask=None):
    m=np.ones(len(L),bool) if mask is None else mask
    X=np.c_[np.ones(m.sum()), N[m]**-a, T[m]**-b, LR[m]**p]
    coef=np.linalg.lstsq(X,L[m],rcond=None)[0]; r=L[m]-X@coef
    return (r**2).sum(), coef, r
best=None
for a in np.linspace(0.30,0.50,41):
  for b in np.linspace(0.28,0.40,49):
    for p in [1.0,1.05,1.1,1.15,1.2,1.25,1.3]:
      s,coef,r=solve(a,b,p)
      if best is None or s<best[0]: best=(s,a,b,p,coef)
s,a,b,p,(E,A,B,c)=best
print('best a=%.4f b=%.4f p=%.2f E=%.4f A=%.1f B=%.1f c=%.4f rms=%.4f'%(a,b,p,E,A,B,c,np.sqrt(s/len(L))))
s,coef,r=solve(a,b,p)
for t in sorted(set(tag)): print('  rms',t,np.sqrt((r[tag==t]**2).mean()), 'mean',r[tag==t].mean())
# residual vs lr for cosine ckpts
m=tag=='ck_cosine'
for lo,hi in [(0,0.15),(0.15,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.01)]:
    mm=m&(LR>=lo)&(LR<hi); print('  lr in',lo,hi,'n',mm.sum(),'mean resid',r[mm].mean().round(4))
np.save('pts.npy',np.c_[N,T,LR,L]); json.dump(list(tag),open('tags.json','w'))
