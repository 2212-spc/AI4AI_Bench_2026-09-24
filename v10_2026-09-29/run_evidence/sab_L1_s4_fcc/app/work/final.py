import csv,glob,numpy as np
from model import erlang_a, replicas
hist=list(csv.DictReader(open('/app/data/history.csv')))
R={}
for r in hist: R[(int(r['day']),int(r['hour']))]=float(r['requests'])
for fn in sorted(glob.glob('/app/data/abtest_*.csv')):
    for r in csv.DictReader(open(fn)):
        if r['fraction_B']!='' : R[(int(r['day']),int(r['hour']))]=float(r['requests_A'])+float(r['requests_B'])
days=sorted(set(d for d,h in R)); full=[d for d in days if all((d,h) in R for h in range(24))]
M=np.array([[R[(d,h)] for h in range(24)] for d in full])
base=M.mean(0); dayf=(M/base).mean(1)
print('days',len(full),'dayf sd',dayf.std())
# per-hour rates: base[h]*dayf (day factor) ; residual per-hour noise ~ poisson-ish, ignore
def hour_score(h,f,p,cap=16):
    rA,rB,gA,gB,th=p['rA'],p['rB'],p['gA'],p['gB'],p['theta']
    m=(1-f)*gA+f*gB; rt=(1-f)*rA+f*rB
    tot=0;w=0
    for df in dayf:
        lam=base[h]*df/3600
        c=replicas(lam*m,cap); ab,W,u=erlang_a(lam,1/m,c,th)
        tot+=lam*((1-ab)*rt-0.008*W); w+=lam
    return tot/w
def sched_value(s,p):
    num=0;den=0
    for h in range(24):
        num+=base[h]*(hour_score(h,s[h],p)-hour_score(h,0.0,p)); den+=base[h]
    return num/den
def optimize(p,grid):
    return [max(grid,key=lambda f:hour_score(h,f,p)) for h in range(24)]
p0=dict(rA=0.63319,rB=0.65836,gA=3.0735,gB=7.2428,theta=0.02177)
se=dict(rA=0.00018,rB=0.00031,gA=0.0017,gB=0.0063,theta=0.00015)
if __name__=='__main__':
    grid=np.round(np.linspace(0,1,101),2)
    s=optimize(p0,grid)
    # refine to 0.005 around
    s2=[]
    for h in range(24):
        g=np.round(np.arange(max(0,s[h]-0.05),min(1,s[h]+0.05)+1e-9,0.005),3)
        s2.append(max(g,key=lambda f:hour_score(h,f,p0)))
    print('sched',s2)
    v2=sched_value(s2,p0); v1=sched_value([1.0]*24,p0)
    print('Q2',v2,'Q1',v1)
    rng=np.random.default_rng(0); Q1=[];Q2=[]
    for i in range(60):
        p={k:p0[k]+se[k]*rng.standard_normal() for k in p0}
        Q1.append(sched_value([1.0]*24,p)); Q2.append(sched_value(s2,p))
    print('Q1 sd',np.std(Q1),'Q2 sd',np.std(Q2))
    # robustness: loss of s2 under perturbed params vs their own optimum
    for k in ['rB','gB','theta']:
        for sgn in [-2,2]:
            p=dict(p0); p[k]+=sgn*se[k]; so=optimize(p,grid)
            print(k,sgn,'regret',sched_value(so,p)-sched_value(s2,p))
    open('/app/rollout.conf','w').write(''.join(f'{h:02d} {s2[h]:.3f}\n' for h in range(24)))
