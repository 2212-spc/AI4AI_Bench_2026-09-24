import json,csv,numpy as np
rows=[]
for r in csv.DictReader(open('/app/notebook/runs.csv')):
 rows.append([float(r[k]) for k in ['N','D','q','sub','loss']])
for line in open('/app/lab_runs.jsonl'):
 r=json.loads(line); rows.append([r['config'][k] for k in ['N','D','q','sub']]+[r['loss']])
a=np.array(rows); n,d,q,sub,y=a.T

def predict(p,n=n,d=d,q=q,sub=sub,mode='exp'):
 floor,A,alpha,B,beta,R,k,power=p
 U=sub*(1-q); e=np.maximum(d/U-1,0)
 if mode=='exp': eff=np.minimum(d,U)+U*R*(-np.expm1(-e/R))
 elif mode=='log': eff=np.minimum(d,U)+U*R*np.log1p(e/R)
 eff*=1+k*q**power
 return floor+A*(n/1e8)**(-alpha)+B*(eff/1e10)**(-beta)

def fit(p,mask=None,mode='exp',fixed={}):
 p=np.array(p,dtype=float); mask=np.ones(len(y),bool) if mask is None else mask
 # N^-small noise as approximation
 weights=(n/1e8)**.15
 def res(p):return ((predict(p,mode=mode)-y)*weights)[mask]
 free=[i for i in range(len(p)) if i not in fixed]
 for i,v in fixed.items():p[i]=v
 damping=1e-3
 for it in range(300):
  r=res(p); jac=[]
  for j in free:
   pp=p.copy(); h=1e-5*(1+abs(p[j]));pp[j]+=h
   jac.append((res(pp)-r)/h)
  J=np.array(jac).T
  step=np.linalg.solve(J.T@J+damping*np.diag(np.maximum(np.diag(J.T@J),1e-4)),-J.T@r)
  pp=p.copy();pp[free]+=step
  if min(pp[1:])<=0 or pp[2]>2 or pp[4]>2 or pp[7]>10:
   damping*=5;continue
  rr=res(pp)
  if rr@rr<r@r:
   p=pp;damping=max(damping/3,1e-10)
   if np.max(abs(step))<1e-8:break
  else: damping*=5
 return p,res(p)@res(p)
if __name__=='__main__':
 for mode in ['exp','log']:
  for fixed in [{7:2},{7:1},{}]:
   p,err=fit([1.8,1.35,.3,.75,.3,6,1,2],mode=mode,fixed=fixed)
   print(mode,fixed,'err',err,'params',p)
   print('residuals',np.round(predict(p,mode=mode)-y,4))
   for s in [2e10,4e10,1e11]:
    f=lambda q:predict(p,n=1e9,d=2e11,q=q,sub=s,mode=mode)
    print('s',s,'diffs',[round(f(q)-f(0),5) for q in [.3,.5,.6,.85]])
   print('rep cost',predict(p,n=1e9,d=2e11,q=0,sub=4e10,mode=mode)-predict(p,n=1e9,d=2e11,q=0,sub=2e11,mode=mode))
   print('fresh .6',predict(p,n=1e9,d=1e11,q=.6,sub=1e12,mode=mode)-predict(p,n=1e9,d=1e11,q=0,sub=1e12,mode=mode))
