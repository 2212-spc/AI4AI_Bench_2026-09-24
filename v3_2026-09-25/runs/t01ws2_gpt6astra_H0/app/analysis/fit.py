import csv,json,numpy as np
from pathlib import Path

def ls(fun,p,maxiter=300):
 p=np.array(p,dtype=float); lam=1e-3
 for it in range(maxiter):
  r=fun(p);cost=r@r
  eps=1e-5
  J=np.column_stack([(fun(p+eps*np.eye(len(p))[j])-fun(p-eps*np.eye(len(p))[j]))/(2*eps) for j in range(len(p))])
  step=np.linalg.solve(J.T@J+lam*np.diag(np.maximum(np.diag(J.T@J),1e-8)), -J.T@r)
  p1=p+step;r1=fun(p1)
  if r1@r1<cost:
   p=p1;lam=max(1e-10,lam/3)
   if np.linalg.norm(step)<1e-9:break
  else:lam=min(1e15,lam*5)
 return p,fun(p)

def load():
 rows=list(csv.DictReader(open('/app/notebook/runs.csv')))
 rows += [dict(r['config'],loss=r['loss']) for r in map(json.loads,open('/app/lab_runs.jsonl')) if r['status']=='ok']
 n,d,lr,y=np.array([[float(r[k]) for k in ['N','D','lr','loss']] for r in rows]).T
 groups=sorted(set(zip(n,d)));idx=np.array([groups.index((a,b)) for a,b in zip(n,d)])
 return n,d,lr,y,groups,idx

if __name__=='__main__':
 n,d,lr,y,groups,idx=load()
 def pred(p):
  x=np.log(lr)-p[0]-p[1]*np.log(n/1e8)-p[2]*np.log(d/1e9)
  return p[5:][idx]+np.where(x<0,p[3],p[4])*x*x
 p,r=ls(lambda p:pred(p)-y,[-5.5,-.15,-.2,.06,.11]+[min(y[idx==i]) for i in range(len(groups))])
 print('lr model',p[:5], 'rmse',np.sqrt(np.mean(r*r)))
 for group,base in zip(groups,p[5:]):print('base',group,base)
 for D in [2e10,2e11,1e12]:print('target lr',D,np.exp(p[0]+p[1]*np.log(10)+p[2]*np.log(D/1e9)))
 np.savez('/app/analysis/fit.npz',p=p,r=r)
