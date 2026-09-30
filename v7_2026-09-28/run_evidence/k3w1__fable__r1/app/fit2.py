import json, numpy as np
from itertools import product
rows=[json.loads(l) for l in open('lab_log.jsonl')]
edge=lambda n: 137*n**-1.571
data=[]
for r in rows:
    a=r['args']; res=r['result']
    if res['diverged'] or res['final_loss']>7: continue
    n,S,eta=a['n'],a['steps'],a['eta']; rr=eta/edge(n)
    if 0.45<=rr<=0.85: data.append((n,S,eta,rr,res['final_loss']))
data=np.array(data); print(len(data),"runs in safe band")
n,S,eta,rr,L=data.T
def model(p,n,S,rr):
    E,A,a,B,b,d=p
    return E+A*n**-a+B*S**-b+d*np.log(rr/0.7)
def loss(p): return np.mean((model(p,n,S,rr)-L)**2)
# crude optimizer: random restarts + Nelder-Mead-ish via scipy? no scipy assumed; do coordinate search
def nm(f,x0,iters=4000):
    x=np.array(x0,float); step=np.abs(x)*0.3+0.05; best=f(x)
    for it in range(iters):
        improved=False
        for i in range(len(x)):
            for s in (+1,-1):
                y=x.copy(); y[i]+=s*step[i]; v=f(y)
                if v<best: x,best,improved=y,v,True
        if not improved: step*=0.5
        if step.max()<1e-7: break
    return x,best
best=None
for a0,b0 in product([0.3,0.6,1.0],[0.2,0.4,0.7]):
    p0=[2.0,30,a0,20,b0,-0.1]
    p,v=nm(loss,p0)
    if best is None or v<best[1]: best=(p,v)
p,v=best; print("params E,A,a,B,b,d",np.round(p,4),"rmse",np.sqrt(v))
for row in data:
    print(row, round(model(p,*row[[0,1,3]]),3), round(row[4]-model(p,*row[[0,1,3]]),3))
C=1006632960000.0
print("== target (r=0.7) ==")
for w in [362,512,724,1024,1448,2048,2896,4096,5793,8192,11585]:
    Sw=C/w**2; print(w,int(Sw),"eta=%.3g"%(0.7*edge(w)),"L=%.3f"%model(p,w,Sw,0.7))
# leave-one-width-out
for drop in [256,362,512,724,1024]:
    m=n!=drop
    f=lambda q: np.mean((model(q,n[m],S[m],rr[m])-L[m])**2)
    q,_=nm(f,p)
    print("drop",drop,np.round(q,3),[(w,round(model(q,w,C/w**2,0.7),3)) for w in [1024,1448,2048,2896,4096,5793]])
