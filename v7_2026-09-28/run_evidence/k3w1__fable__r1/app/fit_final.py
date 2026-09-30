import json, numpy as np
rows=[json.loads(l) for l in open('lab_log.jsonl')]
edge=lambda n: 137*n**-1.571
d=[]
for r in rows:
    a=r['args']; res=r['result']
    if res['diverged'] or res['final_loss']>7: continue
    n,S,eta=a['n'],a['steps'],a['eta']; rr=eta/edge(n)
    if rr<0.35 or rr>0.8: continue
    d.append((n,S,rr,res['final_loss']))
d=np.array(d); n,S,rr,L=d.T
print(len(d),'runs in band')
def model(q,n,S,r):
    E,A,a,B,b,c,r0=q
    return E+A/n**a+B/S**b+c*np.log(r/r0)**2
def nm(f,x0,it=20000):
    x=np.array(x0,float); step=np.abs(x)*0.3+0.05; best=f(x)
    rng=np.random.default_rng(0)
    for i in range(it):
        y=x+step*rng.standard_normal(len(x))*(0.999**i+0.01)
        if y[5]<0 or y[6]<=0: continue
        v=f(y)
        if v<best: best,x=v,y
    return x,best
out={}
for b in [0.25,0.3,0.35,0.4]:
    f=lambda q: np.mean((model(np.r_[q[:4],b,q[4:]],n,S,rr)-L)**2)
    q,v=nm(f,[3.0,400,1.1,8,0.25,1.0])
    q=np.r_[q[:4],b,q[4:]]
    print('b=',b,'rmse',round(v**0.5,4),'params',np.round(q,3))
    B=1.007e12
    for w in [1024,1448,2048]:
        st=B/w**2
        print('   ',w,{r:round(float(model(q,w,st,r)),3) for r in [0.55,0.61,0.7,0.8]})
    print('    1448@2000 r=.54 pred',round(float(model(q,1448,2000,0.54)),3),'obs 4.431')
