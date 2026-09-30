import numpy as np, csv, glob, math
from erlanga import erlang_a
ETA=0.0098; THETA=0.0127; GA=2.4694
DOWS=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
# --- collect all days: requests per (day,hour), dow
req={}; dow={}
for r in csv.DictReader(open('/app/data/history.csv')):
    req.setdefault(int(r['day']),{})[int(r['hour'])]=float(r['requests']); dow[int(r['day'])]=r['dow']
for fn in sorted(glob.glob('/app/data/abtest_*.csv')):
    for r in csv.DictReader(open(fn)):
        d=int(r['day']); req.setdefault(d,{})[int(r['hour'])]=float(r['requests_A'])+float(r['requests_B']); dow[d]=r['dow']
full=[d for d in req if len(req[d])==24]
# iterative profile / multiplier fit
PROF=np.array([np.mean([req[d][h] for d in full]) for h in range(24)])
for it in range(5):
    mult={d:np.mean([req[d][h]/PROF[h] for h in req[d]]) for d in req}
    PROF=np.array([np.mean([req[d][h]/mult[d] for d in full]) for h in range(24)])
    PROF*= 1.0/np.mean([mult[d] for d in full])  # normalize so mean(mult over full days)=1 - arbitrary
mult={d:np.mean([req[d][h]/PROF[h] for h in req[d]]) for d in req}
def dow_fit():
    D={}; res=[]
    for w in DOWS:
        v=np.array([np.log(mult[d]) for d in mult if dow[d]==w]); D[w]=(v.mean(), len(v)); res+=list(v-v.mean())
    res=np.array(res); n=len(res); sig=np.sqrt((res**2).sum()/(n-7))
    return D,sig
D,SIG=dow_fit()
# quadrature over lognormal noise
GH_x,GH_w=np.polynomial.hermite_e.hermegauss(9); GH_w=GH_w/GH_w.sum()
def day_mults(D=D,sig=SIG):
    out=[]  # (mult, weight)
    for w in DOWS:
        for x,wt in zip(GH_x,GH_w): out.append((math.exp(D[w][0]+sig*x), wt/7))
    return out

def hour_score(lam,f,gB,rB,rA,theta=THETA,cap=24):
    g=(1-f)*GA+f*gB
    c=max(4,min(cap,math.ceil(lam*g/0.75-1e-9)))
    pab,EW,u=erlang_a(lam,1/g,c,theta)
    return (1-pab)*((1-f)*rA+f*rB)-ETA*EW
def value(sched,gB,rB,rA,dm=None,theta=THETA):
    dm=dm or day_mults(); tot=0; totA=0; n=0
    for m,wt in dm:
        for h in range(24):
            lam=m*PROF[h]/3600; w=wt*m*PROF[h]
            tot+=w*hour_score(lam,sched[h],gB[h],rB[h],rA[h],theta); totA+=w*hour_score(lam,0.0,gB[h],rB[h],rA[h],theta); n+=w
    return (tot-totA)/n
def optimize(gB,rB,rA,dm=None,theta=THETA,grid=np.linspace(0,1,201)):
    dm=dm or day_mults(); best=np.zeros(24); curves={}
    for h in range(24):
        vals=[]
        for f in grid:
            tot=0;n=0
            for m,wt in dm:
                lam=m*PROF[h]/3600; w=wt*m*PROF[h]; tot+=w*hour_score(lam,f,gB[h],rB[h],rA[h],theta); n+=w
            vals.append(tot/n)
        vals=np.array(vals); best[h]=grid[int(np.argmax(vals))]; curves[h]=vals
    return best,curves
