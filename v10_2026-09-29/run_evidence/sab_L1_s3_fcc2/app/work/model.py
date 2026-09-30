import numpy as np, csv, glob
from erlanga import erlang_a
fl=lambda x: float(x) if x not in ('',None) else np.nan
ETA=0.0112
def load_rates():
    """per-hour list of hourly request counts across all days (history + experiments)"""
    rates={h:[] for h in range(24)}
    for r in csv.DictReader(open('/app/data/history.csv')):
        rates[int(r['hour'])].append(fl(r['requests']))
    for f in sorted(glob.glob('/app/data/abtest_*.csv')):
        for r in csv.DictReader(open(f)):
            rates[int(r['hour'])].append(fl(r['requests_A'])+fl(r['requests_B']))
    return rates
def hour_score(f, lam, P, cap=16):
    m=f*P['mB']+(1-f)*P['mA']
    c=max(4,min(cap,int(np.ceil(lam*m/0.75))))
    pa,ew=erlang_a(lam,1/m,c,P['theta'])
    return (1-pa)*(f*P['rB']+(1-f)*P['rA'])-ETA*ew, pa, ew, c
def value(schedule, P, rates, cap=16):
    """long-run mean score(schedule) - mean score(all A); weighted by requests"""
    tot=0; totA=0; n=0
    for h in range(24):
        for req in rates[h]:
            lam=req/3600
            s,_,_,_=hour_score(schedule[h],lam,P,cap); s0,_,_,_=hour_score(0.0,lam,P,cap)
            tot+=s*req; totA+=s0*req; n+=req
    return (tot-totA)/n
def optimize(P, rates, grid=np.linspace(0,1,101)):
    best=[]
    for h in range(24):
        vals=[]
        for f in grid:
            v=0
            for req in rates[h]:
                s,_,_,_=hour_score(f,req/3600,P); v+=s*req
            vals.append(v)
        best.append(grid[int(np.argmax(vals))])
    return best
P=dict(mA=2.536,mB=5.724,rA=0.6422,rB=0.6702,theta=0.01685)
if __name__=='__main__':
    rates=load_rates()
    sched=optimize(P,rates)
    print("opt sched",[round(x,2) for x in sched])
    print("value opt",value(sched,P,rates))
    print("value allB",value([1]*24,P,rates))
    for h in range(24):
        lam=np.mean(rates[h])/3600
        s,pa,ew,c=hour_score(sched[h],lam,P); s0,pa0,ew0,c0=hour_score(0,lam,P)
        print(f"{h:2d} f={sched[h]:.2f} c={c:2d} util={lam*(sched[h]*P['mB']+(1-sched[h])*P['mA'])/c:.3f} pa={pa:.4f} ew={ew:.2f} gain={s-s0:+.4f}")
