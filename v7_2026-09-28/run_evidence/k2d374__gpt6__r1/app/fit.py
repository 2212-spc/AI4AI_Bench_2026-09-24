import numpy as np,json
DOM=['web','code','math','papers']; EV=['general','code','math']
data=[json.loads(l) for l in open('/app/data.jsonl')]
def lm(fun,p,n=200):
 p=np.array(p,float);lam=.001
 r=fun(p); cost=r@r
 for it in range(n):
  J=np.column_stack([(fun(p+np.eye(len(p))[j]*1e-5)-r)/1e-5 for j in range(len(p))])
  H=J.T@J;g=J.T@r
  try: step=np.linalg.solve(H+lam*np.diag(np.maximum(np.diag(H),1e-6)),g)
  except: break
  q=p-step
  with np.errstate(over='ignore',invalid='ignore',divide='ignore'): rr=fun(q);c=rr@rr
  if np.isfinite(c) and c<cost:
   p=q;r=rr
   if abs(cost-c)<1e-13:break
   cost=c;lam=max(lam/3,1e-10)
  else:lam=min(lam*5,1e15)
 return p,cost
base=[r for r in data if 'pool' not in r]
X=np.array([[r['mix'][d]*r['D']/1e9 for d in DOM] for r in base]);N=np.array([r['N']/5e7 for r in base]);Y=np.array([[r['eval_loss'][e] for e in EV] for r in base])
def predict(p,X,N):
 c,a,alpha,beta=p[:4];w=np.exp(p[4:]);return c+a*N**(-alpha)+(X@w)**(-beta)
if __name__=='__main__':
 pars=[]
 for e in range(3):
  p,c=lm(lambda p:predict(p,X,N)-Y[:,e],[1, .7,.3,.4,*([-2]*4)])
  print(EV[e],p,'rmse',np.sqrt(c/len(X)));pars.append(p)
 np.save('/app/base.npy',pars)
