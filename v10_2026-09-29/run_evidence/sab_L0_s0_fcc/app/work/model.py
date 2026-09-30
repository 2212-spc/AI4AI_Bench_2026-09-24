import csv, numpy as np, collections
from erlanga import erlang_a, replicas
ETA=0.0076
hist=list(csv.DictReader(open('/app/data/history.csv')))
lam_samples=collections.defaultdict(list)
for r in hist: lam_samples[int(r['hour'])].append(float(r['requests']))
def hour_value(h, f, rA, rB, sA, sB, th, cap=32, samples=None):
    """expected (score, queue components) for hour h at fraction f, averaged over arrival samples"""
    samples = samples or lam_samples[h]
    tot=0; totw=0
    for N in samples:
        lam=N/3600; sbar=(1-f)*sA+f*sB
        c=replicas(lam,sbar,cap)
        pab,EW,u=erlang_a(lam,1/sbar,c,th)
        rating=(1-f)*rA+f*rB
        score=rating*(1-pab)-ETA*EW
        tot+=score*N; totw+=N
    return tot/totw
def schedule_value(sched, rA, rB, sA, sB, th, cap=32):
    tot=0; totw=0
    for h in range(24):
        W=sum(lam_samples[h])
        tot+=(hour_value(h,sched[h],rA,rB,sA,sB,th,cap)-hour_value(h,0,rA,rB,sA,sB,th,cap))*W; totw+=W
    return tot/totw
