import numpy as np,json

def fit(fun,p,y,iterations=250):
 p=np.array(p,float); damping=1e-3
 for _ in range(iterations):
  pred=fun(p); r=y-pred
  J=np.column_stack([(fun(p+np.eye(len(p))[j]*1e-5)-fun(p-np.eye(len(p))[j]*1e-5))/2e-5 for j in range(len(p))])
  step=np.linalg.solve(J.T@J+damping*np.diag(np.maximum(np.diag(J.T@J),1e-6)),J.T@r)
  trial=p+step
  if np.all(np.isfinite(fun(trial))) and np.sum((y-fun(trial))**2)<np.sum(r*r):
   p=trial; damping=max(damping/3,1e-10)
   if np.max(np.abs(step))<1e-9: break
  else: damping=min(damping*5,1e10)
 return p,np.sum((y-fun(p))**2)

r=np.genfromtxt('/app/notebook/runs.csv',delimiter=',',names=True,dtype=None,encoding=None)
rows=[[z['N'],z['D'],z['lr'],z['loss']] for z in r]
for z in map(json.loads,open('/app/lab_runs.jsonl')):
 c=z['config']; rows.append([c['N'],c['D'],c['lr'],z['loss']])
N,D,lr,y=np.array(rows).T
coords=np.array([np.log(N/1e8),np.log(D/1e10)])
groups,idx=np.unique(np.array([N,D]).T,axis=0,return_inverse=True)

def penalty(p):
 delta=np.log(lr)-(p[0]+p[1]*coords[0]+p[2]*coords[1])
 return np.where(delta<0,p[3],p[4])*delta**2

def grouped(p):return penalty(p)+p[5:][idx]
p,sse=fit(grouped,[-6.7,-.1,-.2,.05,.12]+[min(y[idx==i]) for i in range(len(groups))],y)
print('grouped fit',p,'rmse',np.sqrt(sse/len(y)))
for i,g in enumerate(groups): print('group',g,'base',p[5+i], 'opt',np.exp(p[0]+p[1]*np.log(g[0]/1e8)+p[2]*np.log(g[1]/1e10)))

def full(p):return penalty(p)+p[5]+np.exp(p[6]-p[7]*coords[0])+np.exp(p[8]-p[9]*coords[1])
f,sse=fit(full,list(p[:5])+[1.7,np.log(1),.3,np.log(.8),.3],y)
print('full fit',f,'rmse',np.sqrt(sse/len(y)))
for pp in [p,f]:
 def opt(n,d):return np.exp(pp[0]+pp[1]*np.log(n/1e8)+pp[2]*np.log(d/1e10))
 print('q1',np.log10(opt(1e9,2e10)),'q2',np.log10(opt(1e9,1e12)),'q3',pp[4]*np.log(.0007182/opt(1e9,1e12))**2,'q4 max',opt(1e9,1e12)*np.sqrt(8),'q5 ratio',10**pp[1], 'q7 opt',opt(1e9,2e11))
print('q8',f[5]+np.exp(f[6]-f[7]*np.log(10))+np.exp(f[8]-f[9]*np.log(100)))
print('residuals',y-full(f))
if __name__=='__main__':np.savez('/app/fit.npz',grouped=p,full=f)
