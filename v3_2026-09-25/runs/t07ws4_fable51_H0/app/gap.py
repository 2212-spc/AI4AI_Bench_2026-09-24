import json, numpy as np
cos=np.load('cos.npy'); wsd=np.load('wsd.npy')
def fitc(data):
    N,D,L=data.T; best=None
    for a in np.linspace(0.3,0.5,81):
        for b in np.linspace(0.25,0.45,81):
            X=np.c_[np.ones_like(N), N**-a, D**-b]
            coef=np.linalg.lstsq(X,L,rcond=None)[0]; s=((L-X@coef)**2).sum()
            if best is None or s<best[0]: best=(s,a,b,coef)
    s,a,b,(E,A,B)=best
    return (lambda N,D: E+A*N**-a+B*D**-b), (a,b,E,A,B)
pc,pcp=fitc(cos); pw,pwp=fitc(wsd); print('cos',pcp,'\nwsd',pwp)
runs=[json.loads(l) for l in open('lab_runs.jsonl')]
nb=json.load(open('notebook/checkpoints.json'))
nbcfg={1:(2e7,2.8e8,'cosine'),2:(5e7,7e8,'cosine'),3:(1.2e8,1.68e9,'cosine'),4:(3e8,4.2e9,'cosine'),5:(1.2e8,1.68e9,'wsd')}
allck=[]
for e in nb:
    N,D,s=nbcfg[e['run']]; allck.append((N,D,s,[(x['frac'],x['loss']) for x in e['checkpoints']]))
for r in runs:
    c=r['config']; allck.append((c['N'],c['D'],c['sched'],[(x['frac'],x['loss']) for x in r['checkpoints']]))
def lr_cos(f): return 0.1+0.9*0.5*(1+np.cos(np.pi*f))
print('\nexcess of ckpt over finished-WSD law at same tokens (cosine runs), then over finished-cosine law')
for N,D,s,cks in allck:
    if s!='cosine': continue
    print(f'N={N:.1e} D={D:.1e} D/N={D/N:.0f}')
    print('  f   ', ' '.join(f'{f:6.2f}' for f,l in cks))
    print('  xW  ', ' '.join(f'{l-pw(N,f*D):6.3f}' for f,l in cks))
    print('  xC  ', ' '.join(f'{l-pc(N,f*D):6.3f}' for f,l in cks))
    print('  x/lr', ' '.join(f'{(l-pw(N,f*D))/(1-lr_cos(f)+1e-9):6.3f}' for f,l in cks))
print('\nWSD runs: excess of ckpt over finished-WSD law at same tokens')
for N,D,s,cks in allck:
    if s!='wsd': continue
    print(f'N={N:.1e} D={D:.1e}')
    print('  f   ', ' '.join(f'{f:6.2f}' for f,l in cks))
    print('  xW  ', ' '.join(f'{l-pw(N,f*D):6.3f}' for f,l in cks))
