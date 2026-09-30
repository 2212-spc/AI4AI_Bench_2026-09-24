import numpy as np, json
from nm import nm
# load all runs from log
runs=[]
for l in open('/app/lab_log.jsonl'):
    r=json.loads(l); a=r['args']; res=r['result']
    if res.get('diverged') or res.get('final_loss') is None or res['final_loss']>20: continue
    runs.append((a['n'],a['steps'],a['eta'],res['final_loss']))
def edge(n): return 0.037*(n/256.0)**-1.40
runs=[x for x in runs if x[2]/edge(x[0])<=0.62 and x[1]>=1000]
n=np.array([x[0] for x in runs],float); S=np.array([x[1] for x in runs],float); eta=np.array([x[2] for x in runs]); L=np.array([x[3] for x in runs])
r=eta/edge(n)
print(len(runs),"runs")
def model(p,n,S,r):
    E,A,a,B,b,c,d = p
    return E + A*n**(-a) + B*(n**c)*(S*(r/0.55)**d)**(-b)
def loss(p): return np.sum((model(p,n,S,r)-L)**2)
best=None
rng=np.random.default_rng(0)
for t in range(40):
    x0=[rng.uniform(1,3),rng.uniform(5,50),rng.uniform(0.2,0.8),rng.uniform(0.2,5),rng.uniform(0.2,0.7),rng.uniform(0.2,1.2),rng.uniform(0.3,1.5)]
    x,fv=nm(loss,x0,iters=5000)
    if best is None or fv<best[1]: best=(x,fv)
p,fv=best
print("params E,A,a,B,b,c,d",np.round(p,4),"rms",np.sqrt(fv/len(L)))
for x,res in zip(runs,model(p,n,S,r)-L): print(x, round(res,4))
C=1.00663296e12
for w in [1024,1448,2048,2896,4096]:
    print(w, [round(model(p,w,C/w**2,rr),3) for rr in (0.4,0.5,0.55,0.6)])
np.save('/app/p2.npy',p)
print("--- ensemble ---")
rng=np.random.default_rng(5); preds=[]
for t in range(60):
    x0=[rng.uniform(1,3),rng.uniform(5,50),rng.uniform(0.2,0.8),rng.uniform(0.2,5),rng.uniform(0.2,0.7),rng.uniform(0.2,1.2),rng.uniform(0.5,1.5)]
    x,fv=nm(loss,x0,iters=3000)
    if fv < 1.5*best[1]+24*0.004**2: preds.append((fv,model(x,2048,C/2048**2,0.53),model(x,2896,C/2896**2,0.55),model(x,1448,C/1448**2,0.55)))
preds.sort()
for q in preds: print(np.round(q,4))
