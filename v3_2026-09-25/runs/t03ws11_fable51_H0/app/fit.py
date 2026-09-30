import json, numpy as np, csv
rows=[]
for l in open('/app/lab_runs.jsonl'):
    r=json.loads(l); c=r['config']
    rows.append((c['N'],c['D'],c['q'],c['sub'],c['seed'],r['loss']))
for r in csv.DictReader(open('/app/notebook/runs.csv')):
    rows.append((float(r['N']),float(r['D']),float(r['q']),float(r['sub']),int(r['seed']),float(r['loss'])))
data=np.array(rows)
N,D,q,sub,seed,L=data.T

def deff(p,N,D,q,sub):
    E,A,al,B,be,Rs,m1,m2=p
    U=sub*(1-q)
    R=np.maximum(D/U-1,0)
    De=np.where(D<=U, D, U*(1+Rs*(1-np.exp(-R/Rs))))
    return De*np.exp(m1*q+m2*q*q)
def model(p,N,D,q,sub):
    E,A,al,B,be,Rs,m1,m2=p
    return E+A*N**(-al)+B*deff(p,N,D,q,sub)**(-be)

def lm(fun,p0,iters=200):
    p=np.array(p0,float); lam=1e-3
    r=fun(p); cost=r@r
    for it in range(iters):
        J=np.zeros((len(r),len(p)))
        for j in range(len(p)):
            h=1e-6*max(abs(p[j]),1e-3); pp=p.copy(); pp[j]+=h
            J[:,j]=(fun(pp)-r)/h
        g=J.T@r; H=J.T@J
        while True:
            step=np.linalg.solve(H+lam*np.diag(np.diag(H)+1e-12),-g)
            pn=p+step; rn=fun(pn); cn=rn@rn
            if cn<cost:
                p,r,cost=pn,rn,cn; lam=max(lam/3,1e-9); break
            lam*=4
            if lam>1e8: return p,r
    return p,r

if __name__=='__main__':
    mask=np.ones(len(L),bool)
    fun=lambda p: model(p,N[mask],D[mask],q[mask],sub[mask])-L[mask]
    p0=[2.0,200,0.4,50,0.3,7,0.5,0.0]
    p,r=lm(fun,p0)
    print('params E,A,al,B,be,Rs,m1,m2:',np.round(p,4))
    print('rms resid',np.sqrt(np.mean(r**2)))
    for i in np.argsort(-abs(r)):
        print(f"N={N[i]:.1e} D={D[i]:.1e} q={q[i]} sub={sub[i]:.2e} s={int(seed[i])} L={L[i]:.4f} res={r[i]:+.4f}")
    np.save('p.npy',p)
