import numpy as np,json
from experiment import U

def load():
 rows=[json.loads(l) for l in open('/app/runs.jsonl')]
 n=np.array([r['N']/5e7 for r in rows]);d=np.array([r['D']/1e9 for r in rows]);w=np.array([r['w'] for r in rows]);u=np.array([r['pool'] if r['pool'] is not None else U for r in rows])/1e9
 y=np.array([[r['result']['eval_loss'][j] for j in ['general','code','math']] for r in rows])
 return rows,(n,d,w,u),y

def predict(p,x):
 n,d,w,u=x
 q=p[:24].reshape(3,8);R=np.exp(p[24:28])
 t=d[:,None]*w
 eff=np.minimum(t,u)+u*R*(-np.expm1(-np.maximum(t/u-1,0)/R))
 s=eff@np.exp(q[:,4:8]).T
 return q[:,0]+np.exp(q[:,1])*n[:,None]**(-np.exp(q[:,2]))+np.maximum(s,1e-12)**(-np.exp(q[:,3]))

def lm(fun,p,steps=150):
 p=p.copy(); lam=.001;r=fun(p); loss=r@r
 for it in range(steps):
  h=1e-5
  J=np.stack([(fun(p+np.eye(len(p))[i]*h)-r)/h for i in range(len(p))],axis=1)
  a=J.T@J;g=J.T@r
  delta=np.linalg.solve(a+lam*np.diag(np.maximum(np.diag(a),1e-6)),g)
  trial=p-delta
  rr=fun(trial);ll=rr@rr
  if np.isfinite(ll) and ll<loss:
   p=trial;r=rr
   if abs(loss-ll)<1e-12:break
   loss=ll;lam=max(lam/3,1e-9)
  else:lam*=5
 return p,loss
if __name__=='__main__':
 rows,x,y=load()
 q=np.zeros((3,8));q[:,0]=[1.8,.7,1.2];q[:,1]=np.log(.6);q[:,2]=np.log(.3);q[:,3]=np.log([.45,.45,.25]);q[:,4:]=np.log([[.5,.2,.04,.3],[.03,1,.08,.2],[.2,.5,1,.35]])
 p=np.r_[q.ravel(),np.log([20,3,2,4])]
 p,loss=lm(lambda p:(predict(p,x)-y).ravel(),p)
 print('loss',loss,'rmse',np.sqrt(loss/y.size));print(p[:24].reshape(3,8));print('R',np.exp(p[24:]));
 pred=predict(p,x)
 for i in np.argsort(np.max(abs(pred-y),axis=1))[-12:]:print(rows[i]['tag'],rows[i]['w'],pred[i]-y[i])
 np.save('/app/model.npy',p)
