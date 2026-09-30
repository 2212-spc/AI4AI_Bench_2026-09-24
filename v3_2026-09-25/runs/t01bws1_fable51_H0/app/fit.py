import numpy as np, csv, json, collections
rows=[(float(r['N']),float(r['D']),float(r['lr']),float(r['loss'])) for r in csv.DictReader(open('notebook/runs.csv'))]
for l in open('lab_runs.jsonl'):
    x=json.loads(l)
    if x.get('status')=='ok':
        c=x['config']; rows.append((c['N'],c['D'],c['lr'],x['loss']))
R=np.array(rows); N,D,LR,L=R.T
lN,lD,lLR=np.log(N),np.log(D),np.log(LR)

def lm(resfun,p0,iters=200,lam=1e-3):
    p=np.array(p0,float)
    r=resfun(p); cost=r@r
    for it in range(iters):
        J=np.zeros((len(r),len(p)))
        for j in range(len(p)):
            dp=np.zeros(len(p)); h=1e-6*max(1,abs(p[j])); dp[j]=h
            J[:,j]=(resfun(p+dp)-r)/h
        g=J.T@r; H=J.T@J
        while True:
            step=np.linalg.solve(H+lam*np.diag(np.diag(H)+1e-12),-g)
            pn=p+step; rn=resfun(pn); cn=rn@rn
            if cn<cost: p,r,cost=pn,rn,cn; lam=max(lam/3,1e-9); break
            lam*=4
            if lam>1e8: return p,r,cost
    return p,r,cost

def model(p,lN,lD,lLR):
    E,A,al,Bc,be,k,b,c,plo,phi=p
    base=E+np.exp(A)*np.exp(-al*lN)+np.exp(Bc)*np.exp(-be*lD)
    x=lLR-(k+b*lN+c*lD)
    pen=np.where(x<0,plo*x**2,phi*x**2)
    return base+pen
def res(p): return model(p,lN,lD,lLR)-L
p0=[2.0,np.log(50),0.3,np.log(50),0.3,0.2,-0.09,-0.23,0.08,0.1]
p,r,cost=lm(res,p0)
names='E lnA al lnB be k b c plo phi'.split()
print({n:round(v,4) for n,v in zip(names,p)}); print('rms',np.sqrt(cost/len(r)))
for row,rr in zip(R,r): print('N=%.1e D=%.1e lr=%.5f L=%.4f res=%+.4f'%(*row,rr))
np.save('p.npy',p)
