import numpy as np, csv
rows=[r for r in csv.DictReader(open('notebook/runs.csv')) if r['qk']=='0' and r['seed']=='0']
N=np.array([float(r['N']) for r in rows]); lr=np.array([float(r['lr']) for r in rows]); L=np.array([float(r['loss']) for r in rows])
sizes=sorted(set(N))
best=None
for g in np.linspace(0.0,0.4,161):
  for le0 in np.linspace(np.log(5e-4),np.log(1e-2),301):
    eta=np.exp(le0)*(N/1e8)**-g
    x=np.log(lr/eta)
    X=np.zeros((len(N),len(sizes)+2))
    for i,s in enumerate(sizes): X[:,i]=(N==s)
    X[:,-2]=np.where(x<0,x**2,0); X[:,-1]=np.where(x>=0,x**2,0)
    coef,res,_,_=np.linalg.lstsq(X,L,rcond=None)
    r=L-X@coef; ss=(r**2).sum()
    if best is None or ss<best[0]: best=(ss,g,np.exp(le0),coef)
ss,g,eta0,coef=best
print("ss",ss,"rms",np.sqrt(ss/len(N)),"g",g,"eta0",eta0,"klo,khi",coef[-2:],"bases",coef[:-2])
print("eta*(7e9)=",eta0*(70)**-g)
for s in sizes: print(s, eta0*(s/1e8)**-g)
