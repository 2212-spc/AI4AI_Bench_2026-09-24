import numpy as np, math
from model import erlangA, replicas
from est import base, dayfactors, Bparams
import est
g,rt,w,mA,rA=Bparams()
DAY=set(range(9,18))
mB=np.array([ (g[9:18]*w[9:18]).sum()/w[9:18].sum() if h in DAY else (g[[x for x in range(24) if x not in DAY]]*w[[x for x in range(24) if x not in DAY]]).sum()/w[[x for x in range(24) if x not in DAY]].sum() for h in range(24)])
rB=np.array([ (rt[9:18]*w[9:18]).sum()/w[9:18].sum() if h in DAY else (rt[[x for x in range(24) if x not in DAY]]*w[[x for x in range(24) if x not in DAY]]).sum()/w[[x for x in range(24) if x not in DAY]].sum() for h in range(24)])
def hour_score(h,f,d,cap=16,theta=0.017124):
    lam=base[h]*d/3600; m=(1-f)*mA+f*mB[h]
    c=replicas(lam,m,cap); pab,W=erlangA(lam,m,c,theta)
    return (1-pab)*((1-f)*rA+f*rB[h])-0.008*W, lam
def day_grid(logsd, n=25):
    # gauss-hermite-ish grid for lognormal with mean 1
    z=np.linspace(-3.5,3.5,n); wts=np.exp(-z*z/2); wts/=wts.sum()
    d=np.exp(logsd*z-logsd**2/2)
    return d,wts
def value(sched, dgrid, theta=0.017124):
    d,wts=dgrid
    num=0; numA=0; den=0
    for h in range(24):
        for di,wi in zip(d,wts):
            s,lam=hour_score(h,sched[h],di,theta=theta); sA,_=hour_score(h,0.0,di,theta=theta)
            num+=wi*lam*s; numA+=wi*lam*sA; den+=wi*lam
    return (num-numA)/den
def hour_obj(h,f,dgrid,theta=0.017124):
    d,wts=dgrid
    return sum(wi*hour_score(h,f,di,theta=theta)[0]*hour_score(h,f,di,theta=theta)[1] for di,wi in zip(d,wts))
def optimize(dgrid, fs=np.linspace(0,1,51), theta=0.017124):
    sched=np.zeros(24); curves=[]
    for h in range(24):
        vals=np.array([hour_obj(h,f,dgrid,theta) for f in fs])
        sched[h]=fs[vals.argmax()]; curves.append(vals)
    return sched, curves
if __name__=='__main__':
    import sys
    logsd=float(sys.argv[1]) if len(sys.argv)>1 else 0.125
    dg=day_grid(logsd)
    sched,curves=optimize(dg)
    print('logsd',logsd)
    print('opt sched', sched.round(2))
    print('Q1 all-B', round(value(np.ones(24),dg),5))
    print('Q2 opt', round(value(sched,dg),5))
    # flatness: per hour, range of f within 0.0005*den/24 of the max
    for h in range(24):
        v=curves[h]; fs=np.linspace(0,1,51); 
        ok=fs[v>=v.max()-0.0002*base[h]/3600*1.0]
        print(h, 'best',sched[h], 'within .0002/req:', ok.min(), ok.max(), 'gain/req', round((v.max()-v[0])/(base[h]/3600),4), 'all-B/req', round((v[-1]-v[0])/(base[h]/3600),4))
