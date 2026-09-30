import numpy as np,json
from pathlib import Path
# get baseline min by config; for notebook fit quadratic min; new fit
rows=[]
R=np.genfromtxt('/app/notebook/runs.csv',delimiter=',',names=True)
for n in np.unique(R['N']):
 a=R[R['N']==n];p=np.polyfit(np.log(a['lr']),a['loss'],2);xm=-p[1]/2/p[0];rows.append((n,20*n,np.polyval(p,xm)))
for line in Path('/app/lab_runs.jsonl').read_text().splitlines():
 a=json.loads(line);c=a['config'];rows.append((c['N'],c['D'],a['loss']))
# unique new configs take only min observed with adjustment not ideal
groups={}
for n,d,y in rows:groups.setdefault((n,d),[]).append(y)
X=np.array([[np.log(n/1e8),np.log(d/2e9),y] for (n,d),ys in groups.items() for y in [min(ys)]])
print(X)
def fit(fun,p):
 p=np.array(p,float);lam=1e-4
 def res(q):return fun(q)
 r=res(p);cost=r@r
 for _ in range(10000):
  J=np.array([(res(p+np.eye(len(p))[j]*1e-5)-res(p-np.eye(len(p))[j]*1e-5))/2e-5 for j in range(len(p))]).T
  H=J.T@J;g=J.T@r
  try:dp=np.linalg.solve(H+lam*np.diag(np.maximum(np.diag(H),1e-10)),g)
  except:break
  q=p-dp;rq=res(q);cc=rq@rq
  if cc<cost:p,r,cost=q,rq,cc;lam=max(lam/2,1e-10)
  else:lam*=3
  if np.linalg.norm(dp)<1e-9:break
 return p,np.sqrt(cost/len(X))
u,v,y=X.T
for form in range(8):
 if form==0:
  # floor + power N + power D
  fn=lambda p:p[0]+p[1]*np.exp(-p[2]*u)+p[3]*np.exp(-p[4]*v)
  init=[1.5,2, .2,2,.2]
 if form==1:
  # floor + power N + power ratio? 
  fn=lambda p:p[0]+p[1]*np.exp(-p[2]*u)+p[3]*np.exp(-p[4]*(v-u))
  init=[1.5,2,.2,2,.2]
 if form==2:
  fn=lambda p:p[0]+p[1]*np.exp(-p[2]*u-p[4]*v)+p[3]*np.exp(-p[5]*v)
  init=[1.5,2,.2,2,.2,.2]
 if form==3:
  fn=lambda p:p[0]+p[1]*np.exp(-p[2]*u)+p[3]*np.exp(-p[4]*v)+p[5]*np.exp(-p[6]*(v-u))
  init=[1.5,2,.2,2,.2,1,.2]
 if form>=4:continue
 p,r=fit(lambda p:fn(p)-y,init)
 target=fn(p).item() if False else fn(p) # array all
 # eval target u=ln10 v=ln500
 uu=np.log(10);vv=np.log(500)
 pp=p
 if form==0:t=pp[0]+pp[1]*np.exp(-pp[2]*uu)+pp[3]*np.exp(-pp[4]*vv)
 if form==1:t=pp[0]+pp[1]*np.exp(-pp[2]*uu)+pp[3]*np.exp(-pp[4]*(vv-uu))
 if form==2:t=pp[0]+pp[1]*np.exp(-pp[2]*uu-pp[4]*vv)+pp[3]*np.exp(-pp[5]*vv)
 if form==3:t=pp[0]+pp[1]*np.exp(-pp[2]*uu)+pp[3]*np.exp(-pp[4]*vv)+pp[5]*np.exp(-pp[6]*(vv-uu))
 print(form,'rms',r,'p',p,'target',t)
