import numpy as np,json

def leastsq(fun,x,maxiter=150):
 x=np.array(x,dtype=float); lam=.001; r=fun(x); s=r@r
 for it in range(maxiter):
  h=1e-5
  J=np.array([(fun(x+np.eye(len(x))[i]*h)-r)/h for i in range(len(x))]).T
  H=J.T@J; g=J.T@r
  try: dx=np.linalg.solve(H+lam*np.diag(np.maximum(np.diag(H),1e-8)), -g)
  except: break
  nr=fun(x+dx); ns=nr@nr
  if np.isfinite(ns) and ns<s:
   x+=dx; r=nr
   if abs(s-ns)<1e-13: break
   s=ns; lam=max(lam/3,1e-10)
  else: lam=min(lam*5,1e12)
 return x,s

def load():
 data=json.load(open('/app/runs.json'))
 N=np.array([r['N']/5e7 for r in data]); T=np.array([[r['D']/1e9*r['mix'][d] for d in ['web','code','math','papers']] for r in data]); Y=np.array([[r['eval_loss'][e] for e in ['general','code','math']] for r in data]); U=np.array([[r.get('pool',{}).get(d,u)/1e9 for d,u in zip(['web','code','math','papers'],[3e12,22468979443.588654,2108934741.0250301,56507653616.707794])] for r in data])
 return N,T,Y,U

def pred(x,N,T):
 c=x[0]; a=np.exp(x[1]); alpha=np.exp(x[2]); beta=np.exp(x[3]); w=np.exp(x[4:8]);
 return c+a*N**(-alpha)+(T@w)**(-beta)
if __name__=='__main__':
 N,T,Y,U=load(); mask=np.arange(len(N))!=17
 ps=[]
 for e in range(3):
  f=lambda x:pred(x,N[mask],T[mask])-Y[mask,e]
  x,s=leastsq(f,[1,np.log(.7),np.log(.35),np.log(.5),-2,-2,-2,-2]); ps.append(x)
  print(e,s,x,'weights',np.exp(x[4:8]),'alpha beta',np.exp(x[2:4]),'c a',x[0],np.exp(x[1])); print('residuals',np.round(pred(x,N,T)-Y[:,e],4))
 np.save('/app/basefit.npy',ps)
