import numpy as np, json,csv
rows=[]
for r in csv.DictReader(open('/app/notebook/runs.csv')): rows.append({k:float(r[k]) for k in ['N','D','q','sub','loss']})
for r in map(json.loads,open('/app/lab_runs.jsonl')): rows.append(dict(r['config'],loss=r['loss']))
x=np.array([[r[k] for k in ['N','D','q','sub']] for r in rows]); y=np.array([r['loss'] for r in rows])
def predict(p,x=x,rep='exp',quality='linear'):
 E,a,alpha,b,beta,R,g=p
 N,D,q,S=x.T; U=S*(1-q); epochs=D/U
 if rep=='exp': de=np.where(epochs<=1,D,U*(1+R*(-np.expm1(-np.maximum(epochs-1,0)/R))))
 elif rep=='power': de=np.where(epochs<=1,D,U*epochs**R)
 elif rep=='log': de=np.where(epochs<=1,D,U*(1+R*np.log1p(np.maximum(epochs-1,0)/R)))
 v=(1+g*q) if quality=='linear' else np.exp(g*q)
 return E+a*(N/1e8)**(-alpha)+b*(de*v/1e10)**(-beta)
def fit(fn,p,y=y):
 p=np.array(p,dtype=float); lam=.01
 for it in range(400):
  f=fn(p); resid=y-f
  steps=1e-5*np.maximum(abs(p),1)
  J=np.column_stack([(fn(p+np.eye(len(p))[j]*steps[j])-f)/steps[j] for j in range(len(p))])
  H=J.T@J
  dp=np.linalg.solve(H+lam*np.diag(np.maximum(np.diag(H),1e-6)),J.T@resid)
  pn=p+dp
  if np.isfinite(fn(pn)).all() and np.sum((y-fn(pn))**2)<sum(resid**2):
   p=pn;lam=max(lam/3,1e-10)
   if np.max(abs(dp))<1e-9:break
  else: lam=min(lam*10,1e12)
 return p,np.sqrt(np.mean((y-fn(p))**2))
def answers(p,rep='exp',quality='linear'):
 def l(S,q,D=2e11): return predict(p,np.array([[1e9,D,q,S]]),rep,quality)[0]
 print('q1',l(4e10,0)-l(1e13,0),'q2',l(4e10,.5)-l(4e10,0))
 print('q3',[(q,l(4e10,q)-l(4e10,0)) for q in [0,.3,.6,.85]])
 print('q4',l(1e13,.6,1e11)-l(1e13,0,1e11))
 print('q5',[(S,l(S,.5)-l(S,0)) for S in [2e10,4e10,1e11]])
 print('q6',[(S,l(S,.85)-l(S,0)) for S in [2e10,4e10,1e11]])
if __name__=='__main__':
 for rep in ['exp','power','log']:
  for quality in ['linear','exp']:
   p,rmse=fit(lambda p:predict(p,rep=rep,quality=quality),[1.7,1,.35,.3,.35,8 if rep!='power' else .7,1])
   print(rep,quality,p,'rmse',rmse);answers(p,rep,quality)
   if rep=='exp' and quality=='linear':
    json.dump(p.tolist(),open('/app/fit.json','w'))
    print('residuals',np.round(y-predict(p),4))
