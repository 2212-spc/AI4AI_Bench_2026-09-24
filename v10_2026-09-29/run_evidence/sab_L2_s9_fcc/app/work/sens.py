from model2 import *
import numpy as np, sys
P=np.load('params.npy',allow_pickle=True).item()
from model import RA
peak=[h for h in range(24) if 9<=h<=17]; off=[h for h in range(24) if h not in peak]
nB=P['nB']; rB=P['rB'].copy(); gB=P['gB'].copy()
for blk in [peak,off]:
    rB[blk]=(rB[blk]*nB[blk]).sum()/nB[blk].sum(); gB[blk]=(gB[blk]*nB[blk]).sum()/nB[blk].sum()
sched=np.load('sched.npy')
def opt_hours(hours,gB,rB,rA,dm,theta,grid):
    s=sched.copy()
    for h in hours:
        vals=[]
        for f in grid:
            tot=0;n=0
            for m,wt in dm:
                lam=m*PROF[h]/3600; w=wt*m*PROF[h]; tot+=w*hour_score(lam,f,gB[h],rB[h],rA[h],theta); n+=w
            vals.append(tot/n)
        s[h]=grid[int(np.argmax(vals))]
    return s
def run(name,theta=THETA,sig=SIG,dshift=0.0,drB=0.0,dgB=0.0,rAoff=0.0):
    D2={w:(D[w][0]+np.log(1+dshift),D[w][1]) for w in DOWS}
    dm=day_mults(D2,sig)
    rB2=rB.copy(); rB2[peak]+=drB; gB2=gB.copy(); gB2[peak]+=dgB; rA2=RA+rAoff
    v=value(sched,gB2,rB2,rA2,dm,theta); vB=value(np.ones(24),gB2,rB2,rA2,dm,theta)
    s2=opt_hours(range(11,18),gB2,rB2,rA2,dm,theta,np.linspace(0.3,1,71)); vo=value(s2,gB2,rB2,rA2,dm,theta)
    print(f'{name:20s} Q2={v:+.5f} Q1={vB:+.5f} opt={vo:+.5f} regret={vo-v:.5f} s11-17={s2[11:18]}',flush=True)
cases={'base':{}, 'theta0.0120':dict(theta=0.0120),'theta0.0134':dict(theta=0.0134),'sig0.03':dict(sig=0.03),'sig0.07':dict(sig=0.07),
 'dow+2%':dict(dshift=0.02),'dow-2%':dict(dshift=-0.02),'rBpeak+0.003':dict(drB=0.003),'rBpeak-0.003':dict(drB=-0.003),
 'gBpeak+0.06':dict(dgB=0.06),'gBpeak-0.06':dict(dgB=-0.06),'rA+0.001':dict(rAoff=0.001)}
for k in sys.argv[1:]: run(k,**cases[k])
