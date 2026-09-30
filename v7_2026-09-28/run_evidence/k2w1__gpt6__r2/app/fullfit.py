from fit import *
U=np.array([3000,40,4,20.])
def arrays(rs):
 n=np.array([r['N']/1e8 for r in rs]);t=np.array([[r['D']/1e9*r['mix'][d] for d in DOM] for r in rs]);u=np.array([[r.get('pool',{}).get(d,U[j]*1e9)/1e9 for j,d in enumerate(DOM)] for r in rs]);y=np.array([[r['eval_loss'][e] for e in EV] for r in rs]);return n,t,u,y

def effective(t,u,tau):
 return np.minimum(t,u)+u*tau*(-np.expm1(-np.maximum(t/u-1,0)/tau))

def predict(p,n,t,u):
 te=effective(t,u,np.exp(p[24:28]));return np.array([pred(p[8*e:8*e+8],n,te) for e in range(3)]).T

def fullfit(rs=None,start=None):
 if rs is None:rs=load()
 n,t,u,y=arrays(rs)
 if start is None:start=np.r_[np.load('base_params.npy').ravel(),np.log([20,2.5,.8,1.5])]
 p,v=lm(lambda p:(predict(p,n,t,u)-y).ravel(),start,maxiter=500)
 return p,v
if __name__=='__main__':
 rs=load();p,v=fullfit(rs);n,t,u,y=arrays(rs)
 print('RMS',np.sqrt(v/y.size),'tau',np.exp(p[24:]))
 for e in range(3):print(EV[e],p[e*8:e*8+4],np.exp(p[e*8+4:e*8+8]))
 print('residuals',np.round(predict(p,n,t,u)-y,4))
 np.save('full_params.npy',p)
