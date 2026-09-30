import numpy as np, csv, math
from model import erlang_a, replicas
rows=list(csv.DictReader(open('/app/data/history.csv')))
R=np.zeros((14,24))
for r in rows: R[int(r['day'])-1,int(r['hour'])]=float(r['requests'])
base=R.mean(0)
dayf=(R/base).mean(1)
def hour_score(h,f,p,cap=16,lams=None):
    rA,rB,gA,gB,th=p['rA'],p['rB'],p['gA'],p['gB'],p['theta']
    m=(1-f)*gA+f*gB; rt=(1-f)*rA+f*rB
    lams=base[h]*dayf/3600 if lams is None else lams
    tot=0;w=0
    for lam in lams:
        c=replicas(lam*m,cap)
        ab,W,u=erlang_a(lam,1/m,c,th)
        tot+=lam*((1-ab)*rt-0.008*W); w+=lam
    return tot/w
def sched_value(sched,p):
    num=0;den=0
    for h in range(24):
        s=hour_score(h,sched[h],p)-hour_score(h,0.0,p)
        num+=base[h]*s; den+=base[h]
    return num/den
def optimize(p,grid=np.linspace(0,1,101)):
    sched=[];vals=[]
    for h in range(24):
        best=max(grid,key=lambda f:hour_score(h,f,p))
        sched.append(best)
    return sched
if __name__=='__main__':
    p=dict(rA=0.6332,rB=0.6582,gA=3.0755,gB=7.211,theta=0.0218)
    s=optimize(p)
    print('opt sched',[round(x,2) for x in s])
    print('value opt',sched_value(s,p))
    print('value allB',sched_value([1]*24,p))
    for h in [14,15,16]:
        print(h,[round(hour_score(h,f,p)-hour_score(h,0,p),4) for f in np.linspace(0,1,11)])
