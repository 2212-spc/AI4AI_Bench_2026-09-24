import numpy as np, csv, glob, json, sys
from model2 import *
import model2
SIGR=0.23
P=np.load('params.npy',allow_pickle=True).item()
nB=P['nB']; rB_raw=P['rB']; gB_raw=P['gB']
peak=[h for h in range(24) if 9<=h<=17]; off=[h for h in range(24) if h not in peak]
# rA: combine history and experiment A samples per hour
hist=list(csv.DictReader(open('/app/data/history.csv')))
nA=np.zeros(24); sA=np.zeros(24)
for r in hist:
    h=int(r['hour']); n=float(r['requests'])*(1-float(r['abandon_rate'])); nA[h]+=n; sA[h]+=n*float(r['mean_rating'])
for fn in sorted(glob.glob('/app/data/abtest_*.csv')):
    for r in csv.DictReader(open(fn)):
        h=int(r['hour']); a=float(r['requests_A'])
        if a>0: n=a*(1-float(r['abandon_rate'])); nA[h]+=n; sA[h]+=n*float(r['rating_A'])
rA_raw=sA/nA; seA=SIGR/np.sqrt(nA)
def shrink(est,se,groups):
    out=est.copy(); outse=se.copy()
    for g in groups:
        w=1/se[g]**2; m=(est[g]*w).sum()/w.sum(); chi=((est[g]-m)**2*w).sum(); df=len(g)-1
        tau2=max(0.0,(chi-df)/w.sum()*len(g)/len(g))  # crude method-of-moments
        # better: tau2 = (chi-df)/(sum w - sum w^2/sum w)
        tau2=max(0.0,(chi-df)/(w.sum()-(w**2).sum()/w.sum()))
        k=tau2/(tau2+se[g]**2); out[g]=m+k*(est[g]-m); outse[g]=np.sqrt(k*se[g]**2+ (1-k)**2/w.sum())
        print(f'group {g[0]}-{g[-1]}: pooled {m:.5f} chi2 {chi:.1f}/{df} tau {np.sqrt(tau2):.5f}',file=sys.stderr)
    return out,outse
seB=SIGR/np.sqrt(nB); segB=gB_raw/np.sqrt(nB)
rB,rBse=shrink(rB_raw,seB,[peak,off])
gB,gBse=shrink(gB_raw,segB,[peak,off])
rA,rAse=shrink(rA_raw,seA,[list(range(24))])
print('rB',rB.round(4),file=sys.stderr); print('gB',gB.round(3),file=sys.stderr); print('rA',rA.round(4),file=sys.stderr)
THETA=0.01270
if 'opt' in sys.argv:
    sched,curves=optimize(gB,rB,rA,theta=THETA,grid=np.linspace(0,1,101))
    # round to 0.01 already; report flatness
    np.save('sched_final.npy',sched)
    print('sched',sched)
    v=value(sched,gB,rB,rA,theta=THETA); vB=value(np.ones(24),gB,rB,rA,theta=THETA)
    print('Q2 est',v,'Q1 est',vB)
    json.dump(dict(sched=list(sched),Q2=v,Q1=vB),open('point.json','w'))
if 'mc' in sys.argv:
    sched=np.load('sched_final.npy'); rng=np.random.default_rng(int(sys.argv[-1]))
    nmc=int(sys.argv[2]); out=[]
    df=sum(D[w][1] for w in DOWS)-7
    for i in range(nmc):
        D2={w:(D[w][0]+rng.normal()*SIG/np.sqrt(D[w][1]),D[w][1]) for w in DOWS}
        sig2=SIG*np.sqrt(rng.chisquare(df)/df)
        dm=day_mults(D2,sig2)
        rB2=rB+rng.normal(size=24)*rBse; gB2=gB+rng.normal(size=24)*gBse; rA2=rA+rng.normal(size=24)*rAse
        th=THETA*(1+0.03*rng.normal())   # queue-model calibration (~3%) folded into theta
        model2.GA=2.4694+0.0014*rng.normal()
        v=value(sched,gB2,rB2,rA2,dm,th); vB=value(np.ones(24),gB2,rB2,rA2,dm,th)
        out.append((v,vB)); print(i,v,vB,flush=True)
    np.save(f'mc_{sys.argv[-1]}.npy',np.array(out))
