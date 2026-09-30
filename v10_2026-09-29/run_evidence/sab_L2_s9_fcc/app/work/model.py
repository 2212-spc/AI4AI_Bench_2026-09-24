import numpy as np, csv, math
from erlanga import erlang_a
ETA=0.0098
hist=list(csv.DictReader(open('/app/data/history.csv')))
M=np.zeros((14,24))
for r in hist: M[int(r['day'])-1,int(r['hour'])]=float(r['requests'])
PROF=M.mean(0)                       # requests per hour, mean day
DAYMULT=(M/PROF).mean(1)             # 14 day multipliers
GA=sum(float(r['mean_gen_time_s'])*float(r['requests']) for r in hist)/M.sum()
RA=np.array([sum(float(r['mean_rating'])*float(r['requests']) for r in hist if int(r['hour'])==h)/M[:,h].sum() for h in range(24)])
THETA=0.0127

def hour_score(lam, f, gB, rB, rA, theta=THETA, cap=24, gA=GA):
    g=(1-f)*gA+f*gB
    c=max(4,min(cap,math.ceil(lam*g/0.75-1e-9)))
    pab,EW,u=erlang_a(lam,1/g,c,theta)
    rating=(1-f)*rA+f*rB
    return (1-pab)*rating-ETA*EW, pab, EW, c

def value(sched, gB, rB, rA=RA, daymult=DAYMULT, theta=THETA, cap=24):
    tot=0; totA=0; n=0
    for m in daymult:
        for h in range(24):
            lam=m*PROF[h]/3600; w=m*PROF[h]
            s,_,_,_=hour_score(lam,sched[h],gB[h],rB[h],rA[h],theta,cap)
            sA,_,_,_=hour_score(lam,0.0,gB[h],rB[h],rA[h],theta,cap)
            tot+=w*s; totA+=w*sA; n+=w
    return (tot-totA)/n

def optimize(gB, rB, rA=RA, daymult=DAYMULT, theta=THETA, grid=np.linspace(0,1,101)):
    best=np.zeros(24)
    for h in range(24):
        vals=[]
        for f in grid:
            tot=0;n=0
            for m in daymult:
                lam=m*PROF[h]/3600; w=m*PROF[h]
                s,_,_,_=hour_score(lam,f,gB[h],rB[h],rA[h],theta)
                tot+=w*s; n+=w
            vals.append(tot/n)
        best[h]=grid[int(np.argmax(vals))]
    return best
