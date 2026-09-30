import numpy as np,json,csv
from pathlib import Path
root=Path('/app')
runs=[json.loads(l) for l in open(root/'lab_runs.jsonl')]
ck={r['run']:r for r in json.load(open(root/'notebook/checkpoints.json'))}
for c in csv.DictReader(open(root/'notebook/runs.csv')):
 runs.append(dict(config={k:float(c[k]) for k in ['N','D']}|{'sched':c['sched']},loss=float(c['loss']),**{k:v for k,v in ck[int(c['run'])].items() if k!='run'}))
rows=[]
def add(n,d,loss,group): rows.append((n,d,loss,group))
def group(s,f):
 if s=='wsd':return 'wsd_stable' if f<=.8 else ('end' if f==1 else 'wsd_'+str(f))
 return 'cos_'+str(float(f))
for r in runs:
 c=r['config'];n,d,s=c['N'],c['D'],c['sched']
 add(n,d,r['loss'],group(s,1))
 for z in r.get('checkpoints') or []: add(n,z['tokens'],z['loss'],group(s,z['frac']))
 for z in r.get('cooldown_branches') or []: add(n,z['tokens'],z['loss'],'end')
groups=sorted(set(r[3] for r in rows)-{'end'})
X=np.array([r[:3] for r in rows]);n,d,y=X.T
G=np.array([[float(r[3]==g) for g in groups] for r in rows])
def fit(wexp=0.3):
 w=(n/1e8)**wexp
 def fun(p):
  a=p[1]*(n/1e8)**-p[2];b=p[3]*(d/1e9)**-p[4]
  f=p[0]+a+b+G@p[5:]
  J=np.column_stack([np.ones(len(n)),a/p[1],-a*np.log(n/1e8),b/p[3],-b*np.log(d/1e9),G])
  return f,J
 p=np.array([1.6,.6,.38,.9,.36]+[.07]*len(groups));lam=1e-6
 for it in range(500):
  f,J=fun(p); J=J*w[:,None];r=(f-y)*w
  step=np.linalg.solve(J.T@J+lam*np.eye(len(p)),-J.T@r)
  new=p+step
  if np.sum(((fun(new)[0]-y)*w)**2)<np.sum(r*r):p=new;lam*=.5
  else:lam*=10
  if np.linalg.norm(step)<1e-10:break
 return p,fun(p)[0]-y
if __name__=='__main__':
 for we in [0,.25,.5]:
  p,res=fit(we);E,A,al,B,be=p[:5];pen=dict(zip(groups,p[5:]));
  print('weights exponent',we,'RMSE',np.sqrt(np.mean(res**2)))
  print('params',p[:5]);print('penalties',pen)
  print('by N',[(N,round(np.std(res[n==N]),5),round(np.mean(res[n==N]),5)) for N in sorted(set(n))])
  def final(N,D):return E+A*(N/1e8)**-al+B*(D/1e9)**-be
  prod=final(3e9,6e10);data=B*60**-be
  print('q1',[(N,final(N,1.08e21/(6*N))) for N in [2.7e8,5.3e8,1.1e9,2.1e9]])
  print('q2',prod,'q3',pen['cos_0.4'])
  print('q4',[(f,data*(f**-be-1)+pen['cos_'+str(f)]-pen['cos_1.0']) for f in [.9,.5]])
  print('q5 difference',data*(.5**-be-.35**-be)+pen['cos_0.5'])
  for g in groups:
   if g.startswith('cos_'): print(g,pen[g])
  np.savez(root/'analysis/fit_result.npz',p=p,res=res)
