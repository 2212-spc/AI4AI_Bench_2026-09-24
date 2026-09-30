import json, numpy as np
rows=[json.loads(l) for l in open('lab_runs.jsonl')]+[json.loads(l) for l in open('notebook/runs.jsonl')]
pts=[]
for r in rows:
    c=r['config']
    if c['spec_g']==0:
        B,L=c['batch'],c['seq']
        t1=r['ms_per_token']; t2=B*1000/r['tokens_per_s']
        pts.append((L,B,t1,t2,c['dur']))
        print(L,B,round(t1,4),round(t2,4),c['dur'])
pts=np.array(pts)
L,B,t1,t2,d=pts.T
T=(t1+t2)/2
from itertools import product
def loss(model,p):
    pred=model(p,L,B)
    if np.any(pred<=0): return 1e9
    w=np.sqrt(d/20)
    return np.sum((w*np.log(T/pred))**2)+np.sum((w*np.log(t1/pred))**2)*0
def m_max(p,L,B):
    w,k,a,b=p; return np.maximum(w+k*B*L/1e5, a*B+b*B*L/1e5)
def m_sum(p,L,B):
    w,k,a,b=p; return w+k*B*L/1e5+a*B+b*B*L/1e5
def m_pnorm(p,L,B):
    w,k,a,b,q=p; m=w+k*B*L/1e5; ar=a*B+b*B*L/1e5; return (m**q+ar**q)**(1/q)
from nm import nelder_mead
for name,model,x0 in [('max',m_max,[6,0.5,0.128,0.0]),('sum',m_sum,[6,0.5,0.05,0.0]),('pnorm',m_pnorm,[6,0.5,0.128,0.0,4])]:
    best=None
    for trial in range(20):
        x=np.array(x0)*(1+0.3*np.random.randn(len(x0)))
        p,v=nelder_mead(lambda p: loss(model,p), x, step=0.2)
        if best is None or v<best[1]: best=(p,v)
    p,v=best
    print(name, np.round(p,5), 'rms%', round(100*np.sqrt(v/len(T)),3))
    print('   resid%', np.round(100*np.log(T/model(p,L,B)),2))
