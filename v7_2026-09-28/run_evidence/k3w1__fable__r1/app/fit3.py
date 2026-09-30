import json, numpy as np
rows=[json.loads(l) for l in open('lab_log.jsonl')]
D=[(r['args']['n'],r['args']['steps'],r['args']['eta'],r['result']['final_loss']) for r in rows if r['op']=='train' and not r['result']['diverged'] and r['result']['final_loss']<20]
D=np.array(D,float); n,S,eta,L=D.T
edge=137*n**-1.571; r=eta/edge
# keep well-tuned band, exclude damaged (r>0.85) and badly undertuned (r<0.4)
m=(r>0.4)&(r<0.85); n,S,eta,L,r=n[m],S[m],eta[m],L[m],r[m]
print(len(L),'runs')
def model(p,n,S,r):
    E,A,a,B,b,c,r0=p
    return E+A/n**a+B/S**b+c*(np.log(r/r0))**2
def nm(f,x0,iters=20000):
    x=np.array(x0,float); step=np.abs(x)*0.3+0.05; best=f(x)
    rng=np.random.default_rng(0)
    for i in range(iters):
        y=x+rng.normal(size=len(x))*step*(0.999**i+0.01)
        v=f(y)
        if v<best: x,best=y,v
    return x,best
def fit(n,S,r,L,bfix=None):
    if bfix is None:
        f=lambda p: np.mean((model(p,n,S,r)-L)**2)+ (1e3 if (p[2]<0 or p[4]<0 or p[6]<=0) else 0)
        p0=[2.0,300,1.0,12,0.3,0.2,0.7]
        p,v=nm(f,p0); return p,np.sqrt(v)
    else:
        f=lambda q: np.mean((model([q[0],q[1],q[2],q[3],bfix,q[4],q[5]],n,S,r)-L)**2)+(1e3 if (q[2]<0 or q[5]<=0) else 0)
        q,v=nm(f,[2.0,300,1.0,12,0.2,0.7]); return [q[0],q[1],q[2],q[3],bfix,q[4],q[5]],np.sqrt(v)
p,rm=fit(n,S,r,L); print('free fit',np.round(p,4),'rmse',round(rm,4))
W=[362,512,724,1024,1448,2048,2896,4096,5793,8192,11585]
def targ(p):
    out=[]
    for w in W:
        St=1.007e12/w**2; ed=137*w**-1.571
        out.append((w,round(ed*p[6],6),round(model(p,w,St,p[6]),3)))
    return out
print(targ(p))
for b in [0.15,0.2,0.25,0.3,0.35,0.4,0.5]:
    q,rm=fit(n,S,r,L,bfix=b); print('b=',b,'rmse',round(rm,4),'E',round(q[0],2),'A',round(q[1],1),'a',round(q[2],3),'B',round(q[3],2),'c',round(q[5],3),'r0',round(q[6],3))
    print('   ',[(w,l) for w,e,l in targ(q)])
