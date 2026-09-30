import csv,json,numpy as np
rows=[]
nb=list(csv.DictReader(open('/app/notebook/runs.csv')))
ck=json.load(open('/app/notebook/checkpoints.json'))
for r,c in zip(nb,ck):
 n,d=float(r['N']),float(r['D']); s=r['sched']
 rows.append((n,d,1.,s,float(r['loss'])))
 for v in c['checkpoints']: rows.append((n,d,v['frac'],s,v['loss']))
for r in map(json.loads,open('/app/lab_runs.jsonl')):
 c=r['config'];n,d,s=c['N'],c['D'],c['sched']
 rows.append((n,d,1.,s,r['loss']))
 for v in r.get('checkpoints',[]): rows.append((n,d,v['frac'],s,v['loss']))
def unpack(rows):
 n=np.array([r[0]/1e8 for r in rows]);d=np.array([r[1]/1e9 for r in rows]);f=np.array([r[2] for r in rows]);s=np.array([r[3]=='cosine' for r in rows]);y=np.array([r[4] for r in rows])
 lr=np.where(s,.55+.45*np.cos(np.pi*f),np.minimum(1,(1-f)*5))
 return n,d,f,s,y,lr
n,d,f,s,y,lr=unpack(rows)
def model(p,rows=rows,mode='constant'):
 n,d,f,s,_,lr=unpack(rows)
 E,A,B,a,b,g=p[:6]
 h=g*np.ones(len(n))
 if mode=='powerlr':lr=lr**p[6]
 if mode=='size':h=h*n**(-p[6])
 if mode=='tokens':h=h*(d*f)**(-p[6])
 if mode=='horizon':h=h*d**(-p[6])
 if mode=='both':h=h*n**(-p[6])*d**(-p[7])
 return E+A*n**(-a)+B*(d*f)**(-b)+h*lr

def fit(mode):
 p=np.array([1.8,1.45,1.8,.36,.35,.30]+([1.] if mode=='powerlr' else [0.] if mode not in ['constant','both'] else [0.,0.] if mode=='both' else []))
 lam=1e-3
 for i in range(400):
  pred=model(p,mode=mode);res=pred-y
  J=np.array([(model(p+np.eye(len(p))[j]*1e-5,mode=mode)-pred)/1e-5 for j in range(len(p))]).T
  step=np.linalg.solve(J.T@J+lam*np.eye(len(p)),-J.T@res)
  q=p+step
  if np.sum((model(q,mode=mode)-y)**2)<np.sum(res**2):
   p=q;lam=max(lam/3,1e-9)
   if np.max(abs(step))<1e-9: break
  else:lam*=5
 return p,model(p,mode=mode)-y
if __name__=='__main__':
 for mode in ['constant','powerlr','size','tokens','horizon','both']:
  p,res=fit(mode)
  print(mode,'params',p,'rmse',np.mean(res**2)**.5)
  targets=[(3e9,6e10,1,'wsd',0),(3e9,6e10,.4,'cosine',0),(3e9,2.4e10,1,'wsd',0),(3e9,6e10,.5,'cosine',0),(3e9,6e10,.9,'cosine',0),(3e9,6e10,1,'cosine',0),(3e9,6e10,.7,'cosine',0)]
  v=model(p,targets,mode)
  print('q2',v[0],'q3',v[1]-v[2],'q4',[v[4]-v[5],v[3]-v[5]],'q5 delta',v[6]-v[2])
  ns=[1.2e9,2.4e9,4.8e9,9.6e9]; opts=[(N,1.8e20/N,1,'wsd',0) for N in ns]
  print('q1',model(p,opts,mode))
  if mode=='constant':
   print('residuals',[(r,round(e,4)) for r,e in zip(rows,res)])
