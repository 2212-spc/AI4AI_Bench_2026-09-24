import numpy as np, csv, glob, math
from model import erlang_a, replicas

ETA=0.0062
def load_arrivals():
    """per-hour list of hourly arrival counts across all days (history + experiments)"""
    arr={h:[] for h in range(24)}
    for r in csv.DictReader(open('/app/data/history.csv')):
        arr[int(r['hour'])].append(float(r['requests']))
    for f in sorted(glob.glob('/app/data/abtest_*.csv')):
        for r in csv.DictReader(open(f)):
            arr[int(r['hour'])].append(float(r['requests_A'])+float(r['requests_B']))
    return arr

def hour_score(lam, f, p, cap=24):
    """expected per-request score at arrival rate lam (per s), fraction f to B"""
    mA,mB,rA,rB,theta=p['mA'],p['mB'],p['rA'],p['rB'],p['theta']
    mg=(1-f)*mA+f*mB
    c=replicas(lam,mg,cap)
    Pab,EW,_=erlang_a(lam,1/mg,c,theta)
    return (1-Pab)*((1-f)*rA+f*rB) - ETA*EW

def lam_grid(arr_h, n=15):
    """quantile grid of hourly rates (per s) from empirical mean/sd; requests count includes Poisson noise,
    the rate variation is what matters for the autoscaler; approximate with normal of rate."""
    m=np.mean(arr_h); s=np.std(arr_h, ddof=1)
    s_rate=math.sqrt(max(s*s-m,0.0))  # remove Poisson component
    from math import erf
    # Gauss-Hermite-like via quantiles
    qs=(np.arange(n)+0.5)/n
    import statistics
    z=np.array([statistics.NormalDist().inv_cdf(q) for q in qs])
    return (m+z*s_rate)/3600.0, m

def schedule_value(sched, p, arr, cap=24, n=15):
    """long-run mean score under sched minus all-A; request-weighted over hours"""
    tot=0; totA=0; W=0
    for h in range(24):
        lams,m=lam_grid(arr[h],n)
        sc=np.mean([hour_score(l,sched[h],p,cap) for l in lams])
        scA=np.mean([hour_score(l,0.0,p,cap) for l in lams])
        # weight: expected requests (per rate) -> approx m * (weights equal)
        tot+=m*sc; totA+=m*scA; W+=m
    return (tot-totA)/W

def best_schedule(p, arr, cap=24, grid=np.linspace(0,1,101), n=15):
    sched=[]; details=[]
    for h in range(24):
        lams,m=lam_grid(arr[h],n)
        vals=[np.mean([hour_score(l,f,p,cap) for l in lams]) for f in grid]
        i=int(np.argmax(vals)); sched.append(grid[i]); details.append((h,grid[i],vals[i],vals[0],vals[-1]))
    return sched, details

P=dict(mA=3.57,mB=8.0,rA=0.6623,rB=0.7042,theta=0.0419)
if __name__=="__main__":
    arr=load_arrivals()
    sched,det=best_schedule(P,arr)
    for d in det: print('%02d f*=%.2f  best=%.4f allA=%.4f allB=%.4f  gain=%.4f'%(d[0],d[1],d[2],d[3],d[4],d[2]-d[3]))
    print('value best', schedule_value(sched,P,arr))
    print('value allB', schedule_value([1.0]*24,P,arr))
