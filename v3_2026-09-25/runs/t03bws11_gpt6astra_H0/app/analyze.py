import numpy as np, json,csv
rows=list(csv.DictReader(open('/app/notebook/runs.csv')))
rows=[{k:float(r[k]) for k in ['N','D','q','sub','loss']} for r in rows]
for r in map(json.loads,open('/app/lab_runs.jsonl')):
 rows.append({**r['config'],'loss':r['loss']})

def fit(fun,x,y,steps=500):
 x=np.array(x,dtype=float); lam=1e-3
 for _ in range(steps):
  f=fun(x); r=f-y
  J=np.column_stack([(fun(x+np.eye(len(x))[i]*1e-5)-fun(x-np.eye(len(x))[i]*1e-5))/2e-5 for i in range(len(x))])
  h=np.linalg.solve(J.T@J+lam*np.eye(len(x)), -J.T@r)
  if np.linalg.norm(h)<1e-9:break
  if np.sum((fun(x+h)-y)**2)<np.sum(r*r):x+=h;lam=max(lam/3,1e-12)
  else:lam=min(lam*5,1e12)
 return x,np.sum((fun(x)-y)**2)

fresh=[r for r in rows if r['q']==0 and r['D']<=r['sub']]
N=np.array([r['N'] for r in fresh]);D=np.array([r['D'] for r in fresh]);y=np.array([r['loss'] for r in fresh])
def law(x,N,D):
 E,A,al,C,be=x
 return E+A*(N/5e7)**(-al)+C*(D/2e9)**(-be)
x,err=fit(lambda x:law(x,N,D),[1.8,1.5,.3,1.2,.3],y)
print('fresh E,A,alpha,C,beta',x,'rms',np.sqrt(err/len(y)))
for r in rows:
 print('q',r['q'],'D',r['D'],'epochs',r['D']/r['sub']/(1-r['q']),'loss',r['loss'],'delta',r['loss']-law(x,r['N'],r['D']))
# fit repetition exponential effective tokens
rep=[r for r in rows if r['q']==0]
N=np.array([r['N'] for r in rep]);D=np.array([r['D'] for r in rep]);U=np.array([r['sub'] for r in rep]);y=np.array([r['loss'] for r in rep])
def effective(D,U,R):
 return np.where(D<=U,D,U*(1+R*(-np.expm1(-np.maximum(D/U-1,0)/R))))
def allrep(p):return law(p[:5],N,effective(D,U,np.exp(p[5])))
p,err=fit(allrep,[*x,np.log(10)],y)
print('REPETITION',p[:5],'R',np.exp(p[5]),'rms',np.sqrt(err/len(y)))
for r in rows:
 z=law(p[:5],r['N'],effective(r['D'],r['sub']*(1-r['q']),np.exp(p[5])))
 print('q',r['q'],'D',r['D'],'epochs',r['D']/r['sub']/(1-r['q']),'loss',r['loss'],'delta from rep',r['loss']-z)

N=np.array([r['N'] for r in rows]);D=np.array([r['D'] for r in rows]);U=np.array([r['sub'] for r in rows]);Q=np.array([r['q'] for r in rows]);y=np.array([r['loss'] for r in rows])

def predict(p,N,D,U,Q,rep_kind='exp',qual_kind='power'):
 # p = E,A,alpha,C,beta, log(R), log(K), power
 E,A,al,C,be,lr,lk,power=p
 R=np.exp(lr); K=np.exp(lk)
 U=U*(1-Q)
 ex=np.maximum(D/U-1,0)
 if rep_kind=='exp':dt=np.where(D<=U,D,U*(1+R*(-np.expm1(-ex/R))))
 elif rep_kind=='rational':dt=np.where(D<=U,D,U*(1+ex/(1+ex/R)))
 elif rep_kind=='log':dt=np.where(D<=U,D,U*(1+R*np.log1p(ex/R)))
 elif rep_kind=='power':dt=np.where(D<=U,D,U*(D/U)**(1/(1+R)))
 if qual_kind=='power':dt=dt*(1+K*Q**power)
 elif qual_kind=='exp':dt=dt*np.exp(K*Q**power)
 elif qual_kind=='addloss':return law(p[:5],N,dt)-K*Q**power
 return law(p[:5],N,dt)

models=[]
for rep_kind in ['exp','rational','log','power']:
 for qual_kind in ['power','exp','addloss']:
  fun=lambda p:predict(p,N,D,U,Q,rep_kind,qual_kind)
  pp,sse=fit(fun,[*x,np.log(7),np.log(1.4),2],y)
  print('\nFULL MODEL',rep_kind,qual_kind,'PARAMS',pp,'R',np.exp(pp[5]),'K',np.exp(pp[6]),'rms',np.sqrt(sse/len(y)))
  print('residuals',np.round((fun(pp)-y)*1000,2))
  model=lambda n,d,u,q:predict(pp,n,d,u,q,rep_kind,qual_kind)
  print('q1',model(1e9,2e11,4e10,0)-model(1e9,2e11,2e12,0))
  print('q2',model(1e9,2e11,4e10,.5)-model(1e9,2e11,4e10,0))
  print('q3',[model(1e9,2e11,4e10,q)-model(1e9,2e11,4e10,0) for q in [0,.3,.6,.85]])
  print('q4',model(1e9,1e11,2e12,.6)-model(1e9,1e11,2e12,0))
  print('q5',[model(1e9,2e11,u,.5)-model(1e9,2e11,u,0) for u in [2e10,4e10,1e11]])
  print('q6',[model(1e9,2e11,u,.6)-model(1e9,2e11,u,0) for u in [2e10,4e10,1e11]])
  models.append(dict(repetition=rep_kind,quality=qual_kind,parameters=pp.tolist(),sse=sse))
json.dump(models,open('/app/fits.json','w'),indent=2)
