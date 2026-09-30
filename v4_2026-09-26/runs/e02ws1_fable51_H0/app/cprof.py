import json, numpy as np
from nm import nelder_mead
rows=[json.loads(l) for l in open('lab_runs.jsonl')]+[json.loads(l) for l in open('notebook/runs.jsonl')]
p=np.load('params.npy'); w,k,ar,br,c0,a0,rho0=p
def m_max(q,L,B): return np.maximum(q[0]+q[1]*B*L/1e5, q[2]*B+q[3]*B*L/1e5)
def E_tokens(a,rho,g,rt=1.0,alt=False):
    if alt: pp=[a*rho**i if i<=4 else a*rho**4*rt**(i-4) for i in range(g)]
    else: pp=[a*rho**i if i<=3 else a*rho**3*rt**(i-3) for i in range(g)]
    return 1+np.cumprod(pp).sum()
# per-run c estimates
print('per-run c estimates (from ms_per_token, from tokens_per_s):')
for r in rows:
    cf=r['config']; B,L,g,d=cf['batch'],cf['seq'],cf['spec_g'],cf['dur']
    if g>0:
        T0=m_max(p[:4],L,B); E=E_tokens(a0,rho0,g)
        s1=r['ms_per_token']*E; s2=B*E*1000/r['tokens_per_s']
        print(L,B,g,d, round((s1/T0-1)/g,4), round((s2/T0-1)/g,4))
def chi2(q, sd_t=0.0145, sd_acc=0.007):
    w,k,ar,br,c,a,rho=q; tot=0
    for r in rows:
        cf=r['config']; B,L,g,d=cf['batch'],cf['seq'],cf['spec_g'],cf['dur']
        T0=m_max((w,k,ar,br),L,B); s=np.sqrt(20/d); step=T0*(1+c*g); E=E_tokens(a,rho,g)
        tot+=(np.log(r['ms_per_token']/(step/E))/(sd_t*s))**2+(np.log(r['tokens_per_s']/(B*E*1000/step))/(sd_t*s))**2
        if g>0:
            for i,qq in enumerate(r['accept_by_position']): tot+=((qq-a*rho**i)/(sd_acc*s))**2
    return tot
base=chi2(p)
print('profile of c:')
for c in [0.046,0.048,0.050,0.052,0.054,0.0557,0.058,0.060,0.062]:
    best=None
    for t in range(6):
        x=np.array([w,k,ar,br,a0,rho0])*(1+0.02*np.random.randn(6))
        f=lambda y: chi2([y[0],y[1],y[2],y[3],c,y[4],y[5]])
        q,v=nelder_mead(f,x,step=0.05,iters=4000)
        if best is None or v<best[1]: best=(q,v)
    print(c, 'dchi2', round(best[1]-base,2))
def gain(a,rho,rt,cc,alt=False): return (E_tokens(a,rho,8,rt,alt)/(1+8*cc))/(E_tokens(a,rho,4)/(1+4*cc))-1
print('gain main interp rt=.85 at c=0.0557,0.050,0.048:', [round(gain(a0,rho0,0.85,cc),4) for cc in (0.0557,0.05,0.048)])
print('gain alt  interp rt=.85 at c=0.0557:', round(gain(a0,rho0,0.85,0.0557,True),4))
print('gain alt  interp rt=1  at c=0.0557:', round(gain(a0,rho0,1.0,0.0557,True),4))
