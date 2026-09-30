import json, numpy as np
from nm import nelder_mead
rows=[json.loads(l) for l in open('lab_runs.jsonl')]+[json.loads(l) for l in open('notebook/runs.jsonl')]
def m_max(q,L,B): return np.maximum(q[0]+q[1]*B*L/1e5, q[2]*B+q[3]*B*L/1e5)
def E_tokens(a,rho,g,rt=1.0):
    pp=[a*rho**i if i<=3 else a*rho**3*rt**(i-3) for i in range(g)]
    return 1+np.cumprod(pp).sum()
def sd_acc(pi,d): return np.sqrt(pi*(1-pi)/(2700*d/20))
def chi2(q, sd_t=0.0145):
    w,k,ar,br,c,a,rho=q; tot=0
    for r in rows:
        cf=r['config']; B,L,g,d=cf['batch'],cf['seq'],cf['spec_g'],cf['dur']
        T0=m_max((w,k,ar,br),L,B); s=np.sqrt(20/d); step=T0*(1+c*g); E=E_tokens(a,rho,g)
        tot+=(np.log(r['ms_per_token']/(step/E))/(sd_t*s))**2+(np.log(r['tokens_per_s']/(B*E*1000/step))/(sd_t*s))**2
        if g>0:
            for i,qq in enumerate(r['accept_by_position']):
                pi=a*rho**i; tot+=((qq-pi)/sd_acc(pi,d))**2
    return tot
x0=[6.06,5.0,0.1247,0.35,0.0557,0.909,0.9467]
best=None
for t in range(30):
    x=np.array(x0)*(1+0.03*np.random.randn(7))
    q,v=nelder_mead(chi2,x,step=0.05,iters=8000)
    if best is None or v<best[1]: best=(q,v)
p,v=best; w,k,ar,br,c,a,rho=p
print('fit',np.round(p,6),'chi2',round(v,1))
np.save('params_final.npy',p)
# direct averages at endpoints (log-mean weighted by dur)
for L,B in [(1446,53),(883,88)]:
    num=0;den=0
    for r in rows:
        cf=r['config']
        if cf['spec_g']==0 and cf['seq']==L and cf['batch']==B:
            d=cf['dur']; num+=d*(np.log(r['ms_per_token'])+np.log(B*1000/r['tokens_per_s'])); den+=2*d
    Tdir=np.exp(num/den); Tmod=m_max(p[:4],L,B)
    print(L,B,'direct T0',round(Tdir,4),'tps',round(B*1000/Tdir,2),'| model T0',round(Tmod,4),'tps',round(B*1000/Tmod,2),'| total dur',den/2)
print('q1 fraction',4*c/(1+4*c))
for rt in (0.85,1.0):
    print('gain rt',rt, (E_tokens(a,rho,8,rt)/(1+8*c))/(E_tokens(a,rho,4)/(1+4*c))-1)
# passing-region profile of gains
n=sum(2+ (len(r['accept_by_position']) if r['config']['spec_g']>0 else 0) for r in rows)
qthr=n*(1-2/(9*n)+3.0902*np.sqrt(2/(9*n)))**3
print('n obs',n,'thr',round(qthr,1),'base chi2',round(v,1))
def zs(aa,rr):
    out=[]
    for r in rows:
        cf=r['config']; B,L,g,d=cf['batch'],cf['seq'],cf['spec_g'],cf['dur']
        T0=m_max(p[:4],L,B); s=np.sqrt(20/d); step=T0*(1+c*g); E=E_tokens(aa,rr,g)
        out+= [np.log(r['ms_per_token']/(step/E))/(0.0145*s), np.log(r['tokens_per_s']/(B*E*1000/step))/(0.0145*s)]
        if g>0:
            for i,qq in enumerate(r['accept_by_position']):
                pi=aa*rr**i; out.append((qq-pi)/sd_acc(pi,d))
    return np.array(out)
hi=(-1,None); lo=(9,None)
for aa in np.linspace(0.8943,0.9547,150):
    for rr in np.linspace(0.9366,0.9735,150):
        z=zs(aa,rr)
        if np.max(np.abs(z))<=3 and np.sum(z**2)<=qthr:
            g1=(E_tokens(aa,rr,8,1.0)/(1+8*c))/(E_tokens(aa,rr,4)/(1+4*c))-1
            g0=(E_tokens(aa,rr,8,0.85)/(1+8*c))/(E_tokens(aa,rr,4)/(1+4*c))-1
            if g1>hi[0]: hi=(g1,(aa,rr))
            if g0<lo[0]: lo=(g0,(aa,rr))
print('max gain rt=1 in passing region',hi); print('min gain rt=.85 in passing region',lo)
z=zs(a,rho); print('witness at fit: max|z|',round(np.max(np.abs(z)),2),'chi2',round(np.sum(z**2),1))
