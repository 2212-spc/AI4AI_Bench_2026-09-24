import json, numpy as np
from nm import nelder_mead
rows=[json.loads(l) for l in open('lab_runs.jsonl')]+[json.loads(l) for l in open('notebook/runs.jsonl')]
p=np.load('params.npy'); w,k,ar,br,c,a0,rho0=p
def m_max(L,B): return max(w+k*B*L/1e5, ar*B+br*B*L/1e5)
def E_tokens(a,rho,g,rt=1.0):
    pp=[a*rho**i if i<=3 else a*rho**3*rt**(i-3) for i in range(g)]
    return 1+np.cumprod(pp).sum()
# per-position acceptance residuals
res={i:[] for i in range(4)}
for r in rows:
    cf=r['config']; g,d=cf['spec_g'],cf['dur']
    if g>0:
        for i,q in enumerate(r['accept_by_position']): res[i].append(np.sqrt(d/20)*(q-a0*rho0**i))
for i in range(4): print('pos',i,'rms(dur20)',round(np.sqrt(np.mean(np.square(res[i]))),5),'n',len(res[i]), 'binom sd n=?')
# time residuals by dur / type
def zs(a,rho,sd_t=0.0145,sd_acc=0.007):
    out=[]
    for r in rows:
        cf=r['config']; B,L,g,d=cf['batch'],cf['seq'],cf['spec_g'],cf['dur']
        T0=m_max(L,B); s=np.sqrt(20/d); step=T0*(1+c*g); E=E_tokens(a,rho,g)
        out.append(np.log(r['ms_per_token']/(step/E))/(sd_t*s))
        out.append(np.log(r['tokens_per_s']/(B*E*1000/step))/(sd_t*s))
        if g>0:
            for i,q in enumerate(r['accept_by_position']): out.append((q-a*rho**i)/(sd_acc*s))
    return np.array(out)
z=zs(a0,rho0); print('n obs',len(z),'chi2',round(np.sum(z**2),1),'max|z|',round(np.max(np.abs(z)),2))
# chi2 0.999 quantile approx (Wilson-Hilferty)
n=len(z); q=n*(1-2/(9*n)+3.0902*np.sqrt(2/(9*n)))**3; print('chi2 .999 quantile ~',round(q,1))
def gain(a,rho,rt,cc=c): return (E_tokens(a,rho,8,rt)/(1+8*cc))/(E_tokens(a,rho,4)/(1+4*cc))-1
# grid over a,rho within declared ranges
A=np.linspace(0.8943,0.9547,200); R=np.linspace(0.9366,0.9735,200)
best_hi=(-1,None); best_lo=(9,None)
for a in A:
    for rho in R:
        z=zs(a,rho); 
        if np.max(np.abs(z))<=3 and np.sum(z**2)<=q:
            gh=gain(a,rho,1.0); gl=gain(a,rho,0.85)
            if gh>best_hi[0]: best_hi=(gh,(a,rho))
            if gl<best_lo[0]: best_lo=(gl,(a,rho))
print('max gain rt=1 over passing region:',best_hi)
print('min gain rt=0.85 over passing region:',best_lo)
print('central gains', gain(a0,rho0,1.0), gain(a0,rho0,0.85))
