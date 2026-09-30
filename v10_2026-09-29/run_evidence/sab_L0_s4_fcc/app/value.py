from model import *
import csv, numpy as np, sys
rA=0.64027; rB=0.68304; gA=1.948; gB=4.0837; theta=0.01816
# empirical hourly arrival counts from history + experiments
counts={h:[] for h in range(24)}
for r in csv.DictReader(open('data/history.csv')): counts[int(r['hour'])].append(float(r['requests']))
for f in ['data/abtest_000.csv','data/abtest_001.csv','data/abtest_003.csv']:
    for r in csv.DictReader(open(f)): counts[int(r['hour'])].append(float(r['requests_A'])+float(r['requests_B']))

def hour_score(h, f, rA=rA, rB=rB, gA=gA, gB=gB, theta=theta, cap=32):
    tot=0; n=0
    for req in counts[h]:
        lam=req/3600
        c,Pab,EW,u=mix_model(lam,f,gA,gB,theta,cap)
        rat=(1-f)*rA+f*rB
        tot+=req*(rat*(1-Pab)-0.0069*EW); n+=req
    return tot/n, n

def sched_value(s, **kw):
    tot=0; n=0; tot0=0
    for h in range(24):
        v,nh=hour_score(h,s[h],**kw); v0,_=hour_score(h,0.0,**kw)
        tot+=v*nh; tot0+=v0*nh; n+=nh
    return (tot-tot0)/n

if __name__=="__main__":
    print("all-B value: %.5f"%sched_value([1.0]*24))
    # per-hour best fraction on grid
    best=[]
    for h in range(24):
        grid=np.linspace(0,1,21)
        vals=[hour_score(h,f)[0] for f in grid]
        best.append(grid[int(np.argmax(vals))])
    print("best per-hour fractions:", best)
    print("value of best:", sched_value(best))
    # sensitivity
    for kw in [dict(theta=0.014),dict(theta=0.023),dict(gB=4.3),dict(gA=1.99),dict(rB=rB+0.0007),dict(rB=rB-0.0007),dict(rA=rA+0.0005)]:
        print(kw, "all-B value %.5f"%sched_value([1.0]*24,**kw))
