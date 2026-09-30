import numpy as np, json
from pathlib import Path
r=np.genfromtxt('/app/notebook/runs.csv',delimiter=',',names=True,dtype=None,encoding=None)
data=[[a['N'],a['D'],a['lr'],a['loss']] for a in r]
for line in Path('/app/lab_runs.jsonl').read_text().splitlines():
 a=json.loads(line);c=a['config'];data.append([c['N'],c['D'],c['lr'],a['loss']])
data=np.array(data);N,D,lr,y=data.T
pairs=list(dict.fromkeys(zip(N,D))); group=np.array([pairs.index((n,d)) for n,d in zip(N,D)])
u=np.log(N/1e8);v=np.log(D/2e9);x=np.log(lr)

def lm(fun,p,steps=200):
 p=np.array(p,dtype=float);lam=.001;r=fun(p);cost=r@r
 for it in range(steps):
  J=np.array([(fun(p+np.eye(len(p))[j]*1e-5)-fun(p-np.eye(len(p))[j]*1e-5))/2e-5 for j in range(len(p))]).T
  H=J.T@J;g=J.T@r
  try: dp=np.linalg.solve(H+lam*np.diag(np.maximum(np.diag(H),1e-8)),g)
  except np.linalg.LinAlgError: break
  q=p-dp; rq=fun(q);cq=rq@rq
  if np.isfinite(cq) and cq<cost:
   p=q;r=rq;cost=cq;lam=max(lam/3,1e-10)
   if np.linalg.norm(dp)<1e-8: break
  else: lam*=5
  if lam>1e15:break
 return p,cost

def penalty(z,p,kind):
 if kind=='quad':return p[0]*z*z
 if kind=='piece':return p[0]*z*z+p[1]*np.maximum(z,0)**2
 if kind=='cubic':return p[0]*z*z+p[1]*z**3
 if kind=='quartic':return p[0]*z*z+p[1]*z**3+p[2]*z**4
 if kind=='poscubic':return p[0]*z*z+p[1]*np.maximum(z,0)**3
 if kind=='tanh':return p[0]*z*z*(1+p[1]*np.tanh(z))

if __name__=='__main__':
 for wt in [0,.25,.5]:
  weights=(N/1e8)**wt
  for kind in ['quad','piece','cubic','quartic','poscubic','tanh']:
   npn={'quad':1,'quartic':3}.get(kind,2)
   init=[-5.6,.15,.2]+[.09,.02,.002][:npn]+[y[group==i].min() for i in range(len(pairs))]
   def pred(p):
    z=x-(p[0]-p[1]*u-p[2]*v)
    return p[3+npn:][group]+penalty(z,p[3:3+npn],kind)
   p,c=lm(lambda p:(pred(p)-y)*weights,init)
   print('weight',wt,kind,'rmse',round(np.sqrt(c/len(y)),5),'coeff',np.round(p[:3+npn],6))
   print('  pred log10s',[(d,round((p[0]-p[1]*np.log(10)-p[2]*np.log(d/2e9))/np.log(10),5)) for d in [2e10,2e11,1e12]])
   if kind=='piece' and wt==.25:
    np.savez('/app/fit.npz',p=p,pairs=np.array(pairs),residual=y-pred(p))
    print('  baseline',list(zip(pairs,p[3+npn:])))
