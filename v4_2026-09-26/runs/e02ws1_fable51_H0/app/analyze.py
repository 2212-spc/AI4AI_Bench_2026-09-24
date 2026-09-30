import json, numpy as np
from nm import nelder_mead
rows=[json.loads(l) for l in open('lab_runs.jsonl')]+[json.loads(l) for l in open('notebook/runs.jsonl')]
def m_max(p,L,B):
    w,k,a,b=p; return np.maximum(w+k*B*L/1e5, a*B+b*B*L/1e5)
# acceptance model
spec=[r for r in rows if r['config']['spec_g']>0]
nons=[r for r in rows if r['config']['spec_g']==0]
def E_tokens(a,rho,g,rt=1.0):
    p=[a*rho**i if i<=3 else a*rho**3*rt**(i-3) for i in range(g)]
    cp=np.cumprod(p); return 1+cp.sum()
# joint fit: params w,k,ar,br,c,a,rho ; observables: ms_per_token, tokens_per_s (log-resid, weight sqrt(dur/20)), accept_by_position (resid weight sqrt(dur/20))
def resid(p, sd_t=0.01, sd_acc=0.005):
    w,k,ar,br,c,a,rho=p
    out=[]
    for r in rows:
        cf=r['config']; B,L,g,d=cf['batch'],cf['seq'],cf['spec_g'],cf['dur']
        T0=m_max((w,k,ar,br),L,B); wt=np.sqrt(d/20)
        step=T0*(1+c*g); E=E_tokens(a,rho,g)
        out.append(wt*np.log(r['ms_per_token']/(step/E))/sd_t)
        out.append(wt*np.log(r['tokens_per_s']/(B*E*1000/step))/sd_t)
        if g>0:
            for i,q in enumerate(r['accept_by_position']):
                out.append(wt*(q-a*rho**i)/sd_acc)
    return np.array(out)
f=lambda p: np.sum(resid(p)**2)
x0=[6.01,5.1,0.1247,0.39,0.05,0.905,0.95]
best=None
for t in range(30):
    x=np.array(x0)*(1+0.05*np.random.randn(7))
    p,v=nelder_mead(f,x,step=0.1,iters=8000)
    if best is None or v<best[1]: best=(p,v)
p,v=best
print('params w,k,ar,br,c,a,rho =',np.round(p,6),'chi2',round(v,2))
res=resid(p)
# separate noise estimates
rt=[];ra=[]
for r in rows:
    cf=r['config']; B,L,g,d=cf['batch'],cf['seq'],cf['spec_g'],cf['dur']
    w,k,ar,br,c,a,rho=p
    T0=m_max((w,k,ar,br),L,B); wt=np.sqrt(d/20); step=T0*(1+c*g); E=E_tokens(a,rho,g)
    rt.append(wt*np.log(r['ms_per_token']/(step/E))); rt.append(wt*np.log(r['tokens_per_s']/(B*E*1000/step)))
    if g>0:
        for i,q in enumerate(r['accept_by_position']): ra.append((wt*(q-a*rho**i), i))
rt=np.array(rt); print('time noise rms at dur=20 (%):',round(100*np.sqrt(np.mean(rt**2)),3), 'n',len(rt))
ra=np.array([x for x,_ in ra]); print('acc noise rms at dur=20:',round(np.sqrt(np.mean(ra**2)),5),'n',len(ra))
for i in range(4):
    v_=[x for x,j in [(wt,0)] ] 
np.save('params.npy',p)
w,k,ar,br,c,a,rho=p
print('fraction q1 = 4c/(1+4c) =', 4*c/(1+4*c))
for L,B in [(1446,53),(883,88)]:
    T0=m_max((w,k,ar,br),L,B); print(L,B,'T0',T0,'tps',B*1000/T0, 'mem',w+k*B*L/1e5,'arith',ar*B+br*B*L/1e5)
# scan production throughput over L
pool=15325136008; kvpt=302561794.6814095/1536
vals=[]
for L in range(883,1447):
    B=int(pool//(kvpt*L)); T0=m_max((w,k,ar,br),L,B); vals.append((B*1000/T0,L,B))
print('min',min(vals),'max',max(vals))
for rt_ in [0.85,1.0]:
    R=(E_tokens(a,rho,8,rt_)/(1+8*c))/(E_tokens(a,rho,4)/(1+4*c)); print('rho_tail',rt_,'gain',R-1)
