import numpy as np,json

def lm(fun,x,y,p,maxiter=300):
 p=np.array(p,float); lam=.001
 for _ in range(maxiter):
  r=fun(x,p)-y
  J=np.column_stack([(fun(x,p+np.eye(len(p))[i]*1e-5)-fun(x,p-np.eye(len(p))[i]*1e-5))/2e-5 for i in range(len(p))])
  H=J.T@J
  d=np.linalg.solve(H+lam*np.diag(np.maximum(np.diag(H),1e-6)),J.T@r)
  pn=p-d
  if np.sum((fun(x,pn)-y)**2)<r@r:
   p=pn;lam=max(1e-10,lam/3)
   if np.linalg.norm(d)<1e-9:break
  else:lam*=5
 return p,np.sqrt(np.mean((fun(x,p)-y)**2))

def base(x,p):
 n,s,e=x.T
 c,a,al,b,be=p
 return c+a*(n/512)**(-al)+b*(s*e)**(-be)

def load():
 ds=[json.loads(l) for l in open('/app/observations.jsonl')]
 ds=[d for d in ds if not d['diverged']]
 return np.array([[d['n'],d['steps'],d['eta']] for d in ds]),np.array([d['final_loss'] for d in ds])
if __name__=='__main__':
 x,y=load();n,s,e=x.T
 keep=e<.5*.012*(n/512)**-1.6
 p,err=lm(base,x[keep],y[keep],[2,1.7,.5,3,.42])
 print('base',p,'rmse',err)
 for row, yy, res in zip(x,y,y-base(x,p)):print(row,yy,round(res,4))
 np.save('/app/base.npy',p)
