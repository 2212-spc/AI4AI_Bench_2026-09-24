import numpy as np, json,csv

def fit(fun,p,y,steps=200):
 p=np.array(p,float); lam=1e-3
 for _ in range(steps):
  pred=fun(p); r=pred-y
  jac=np.column_stack([(fun(p+np.eye(len(p))[i]*1e-5)-pred)/1e-5 for i in range(len(p))])
  a=jac.T@jac
  dp=np.linalg.solve(a+lam*np.diag(np.maximum(np.diag(a),1e-8)), -jac.T@r)
  pp=p+dp
  if np.sum((fun(pp)-y)**2)<np.sum(r*r):
   p=pp; lam=max(lam/3,1e-10)
   if np.max(abs(dp))<1e-9:break
  else:lam=min(lam*5,1e12)
 return p,np.sqrt(np.mean((fun(p)-y)**2))

rows=list(csv.DictReader(open('/app/notebook/runs.csv')))
rows=[{k:float(r[k]) for k in ['N','D','lr','loss']} for r in rows]
for l in open('/app/lab_runs.jsonl'):
 r=json.loads(l); rows.append(dict(N=r['config']['N'],D=r['config']['D'],lr=r['config']['lr'],loss=r['loss']))
N,D,lr,y=np.array([[r[k] for k in ['N','D','lr','loss']] for r in rows]).T
n=np.log(N/1e8); d=np.log(D/1e9); x=np.log(lr)
groups=list(dict.fromkeys(zip(N,D))); gi=np.array([groups.index(k) for k in zip(N,D)])

def penalty(p,n=n,d=d,x=x):
 z=x-(p[0]+p[1]*n+p[2]*d)
 return p[3]*np.minimum(z,0)**2+p[4]*np.maximum(z,0)**2

def model(p):return p[5:][gi]+penalty(p)
p,rmse=fit(model,[-6.1,-.1,-.25,.06,.10]+[min(y[gi==i]) for i in range(len(groups))],y)
print('lr parameters',p[:5],'rmse',rmse)
for g,b in zip(groups,p[5:]): print('baseline',g,b)
for Dprod in [2e10,2e11,1e12]:
 opt=p[0]+p[1]*np.log(10)+p[2]*np.log(Dprod/1e9)
 print('production',Dprod,'lr',np.exp(opt),'log10',opt/np.log(10))
opt=p[0]+p[1]*np.log(10)+p[2]*np.log(1000)
print('excess',penalty(p,np.log(10),np.log(1000),np.log(.0007182)))

def full(p):
 return p[5]+p[6]*np.exp(-p[7]*n)+p[8]*np.exp(-p[9]*d)+penalty(p)
pf,rf=fit(full,list(p[:5])+[1.7,1,.3,1.3,.3],y)
print('full fit',pf,'rmse',rf)
print('production loss',pf[5]+pf[6]*10**(-pf[7])+pf[8]*1000**(-pf[9]))
print('residuals max',max(abs(full(pf)-y)))
if __name__=='__main__':
 json.dump(dict(lr=p[:5].tolist(),full=pf.tolist(),rmse=rf),open('/app/fit.json','w'),indent=2)
