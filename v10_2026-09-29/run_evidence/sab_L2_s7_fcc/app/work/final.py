import numpy as np, json, sys, itertools
from model import erlangA, replicas
from est import base, Bparams
g,rt,w,mA,rA=Bparams()
DAY=[h for h in range(24) if 9<=h<=17]; NIGHT=[h for h in range(24) if h not in DAY]
def pooled(arr,idx): return (arr[idx]*w[idx]).sum()/w[idx].sum()
P=dict(mA=mA, rA=rA, mBd=pooled(g,DAY), mBn=pooled(g,NIGHT), rBd=pooled(rt,DAY), rBn=pooled(rt,NIGHT), theta=0.017124)
bydow=json.load(open('bydow.json'))
def day_dist(kind, within_sd=0.0424, logsd=0.1323, nz=9):
    z=np.linspace(-2.8,2.8,nz); wz=np.exp(-z*z/2); wz/=wz.sum()
    if kind=='dow':
        d=[]; wt=[]
        for k,v in bydow.items():
            mu=np.mean(v)
            for zi,wi in zip(z,wz): d.append(mu*np.exp(within_sd*zi-within_sd**2/2)); wt.append(wi/7)
        return np.array(d), np.array(wt)
    else:
        z=np.linspace(-3.5,3.5,31); wz=np.exp(-z*z/2); wz/=wz.sum()
        return np.exp(logsd*z-logsd**2/2), wz
def hour_score(h,f,d,P,cap=16):
    lam=base[h]*d/3600
    mB=P['mBd'] if h in DAY else P['mBn']; rB=P['rBd'] if h in DAY else P['rBn']
    m=(1-f)*P['mA']+f*mB
    c=replicas(lam,m,cap); pab,W=erlangA(lam,m,c,P['theta'])
    return (1-pab)*((1-f)*P['rA']+f*rB)-0.008*W, lam
FS=np.round(np.linspace(0,1,51),2)
def tables(P,dist):
    d,wt=dist
    T=np.zeros((24,len(FS)))   # expected lam*score per hour per f
    den=0
    for h in range(24):
        for di,wi in zip(d,wt):
            den+=wi*base[h]*di/3600
            for j,f in enumerate(FS):
                s,lam=hour_score(h,f,di,P); T[h,j]+=wi*lam*s
    return T,den
def value(T,den,sched):
    idx=[int(round(f*50)) for f in sched]
    return (sum(T[h,idx[h]] for h in range(24))-T[:,0].sum())/den
def optimize(T,den):
    return np.array([FS[T[h].argmax()] for h in range(24)])
if __name__=='__main__':
    kind=sys.argv[1]; tag=sys.argv[2] if len(sys.argv)>2 else ''
    P2=dict(P)
    for a in sys.argv[3:]:
        k,v=a.split('='); P2[k]=float(v)
    kw={}
    if kind.startswith('dow:'): kw['within_sd']=float(kind.split(':')[1]); kind='dow'
    if kind.startswith('logn:'): kw['logsd']=float(kind.split(':')[1]); kind='logn'
    T,den=tables(P2,day_dist(kind,**kw))
    sched=optimize(T,den)
    out=dict(kind=kind,kw=kw,P=P2,sched=sched.tolist(),Q1=value(T,den,np.ones(24)),Q2=value(T,den,sched))
    np.save('T_%s.npy'%tag,T); json.dump(out,open('res_%s.json'%tag,'w'))
    print(json.dumps(out))
