import json, numpy as np
rows=[json.loads(l) for l in open('/app/lab_log.jsonl')]
D=[(r['args']['n'],r['args']['steps'],r['args']['eta'],r['result']['diverged'],r['result']['final_loss']) for r in rows]
# edge model
edge=lambda n: 137*n**-1.571
clean=[(n,s,e,L) for n,s,e,d,L in D if not d and L<7]
# drop spike-damaged points (loss clearly above neighbours): keep eta <= 0.6*edge
fitpts=[(n,s,e,L) for n,s,e,L in clean if e<=0.6*edge(n)]
print(len(clean),len(fitpts))
n=np.array([p[0] for p in fitpts],float); s=np.array([p[1] for p in fitpts],float)
e=np.array([p[2] for p in fitpts]); L=np.array([p[3] for p in fitpts])
x=np.log(e/edge(n))
def model(p,n,s,x):
    E,A,a,B,b,c,x0=p
    return E+A*n**-a+B*(s*np.minimum(1,np.exp(x)/np.exp(x0)))**-b + c*(x-x0)**2*0
def pred(p,n,s,x):
    E,A,a,B,b,c,x0=p
    # effective steps scale with eta below optimum; quadratic penalty around x0
    return E+A*n**-a+B*s**-b*np.exp(-c*(x-x0)) + 0*x
best=None
rng=np.random.default_rng(0)
def loss(p):
    r=pred(p,n,s,x)-L
    if p[2]<=0 or p[4]<=0 or p[1]<0 or p[3]<0: return 1e9
    return np.mean(r**2)
from itertools import product
# crude Nelder-Mead
def nm(f,p0,iters=4000):
    p0=np.array(p0,float); k=len(p0)
    simplex=[p0]+[p0+np.eye(k)[i]*max(abs(p0[i])*0.2,0.05) for i in range(k)]
    vals=[f(p) for p in simplex]
    for _ in range(iters):
        o=np.argsort(vals); simplex=[simplex[i] for i in o]; vals=[vals[i] for i in o]
        c=np.mean(simplex[:-1],0); xr=c+(c-simplex[-1]); fr=f(xr)
        if fr<vals[0]:
            xe=c+2*(c-simplex[-1]); fe=f(xe)
            if fe<fr: simplex[-1],vals[-1]=xe,fe
            else: simplex[-1],vals[-1]=xr,fr
        elif fr<vals[-2]: simplex[-1],vals[-1]=xr,fr
        else:
            xc=c+0.5*(simplex[-1]-c); fc=f(xc)
            if fc<vals[-1]: simplex[-1],vals[-1]=xc,fc
            else:
                simplex=[simplex[0]+0.5*(q-simplex[0]) for q in simplex]; vals=[f(q) for q in simplex]
    return simplex[0],vals[0]
for a0,b0 in product([0.3,0.6,1.0],[0.3,0.5]):
    p,v=nm(loss,[3.0,30,a0,20,b0,0.5,-0.7])
    if best is None or v<best[1]: best=(p,v)
p,v=best
print('params E,A,a,B,b,c,x0',np.round(p,4),'rmse',np.sqrt(v))
for q,r in zip(fitpts,pred(p,n,s,x)): print(q,round(r,3),round(q[3]-r,3))
C=1006632960000.0
print('== target ==')
for w in [362,512,724,1024,1448,2048,2896,4096,5793,8192,11585]:
    S=C/w**2
    for x0 in [p[6],-0.5,-0.9]:
        print(w,int(S),'eta=%.3g'%(edge(w)*np.exp(x0)),'x=%.2f'%x0,'L=%.3f'%pred(p,w,S,x0))
# iso-compute empirical
print('== iso-compute empirical ==')
for n_,s_,e_,L_ in sorted(clean,key=lambda t:(t[0]**2*t[1],t[0])):
    print('%.2e'%(n_**2*s_),n_,s_,e_,L_)
