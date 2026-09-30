from fit import *
WEIGHT=np.array([.237,.36,.403])
def optimize(x):
 rng=np.random.default_rng(4);ms=rng.dirichlet(np.ones(4),10000)
 losses=predict(x,50,ms*500,U)@WEIGHT;m=ms[losses.argmin()];v=losses.min()
 for step in [.05,.02,.01,.005,.002,.001,.0005,.0002,.0001,.00005]:
  for k in range(100):
   moves=np.array([m+step*(np.eye(4)[i]-np.eye(4)[j]) for i in range(4) for j in range(4)])
   moves=moves[np.all(moves>=0,axis=1)]
   ls=predict(x,50,moves*500,U)@WEIGHT;idx=ls.argmin()
   if ls[idx]>=v-1e-12:break
   m=moves[idx];v=ls[idx]
 return m,v
if __name__=='__main__':
 x=np.load('/app/fit.npy');m,v=optimize(x)
 print('optimal',m,'loss',v,'eval',predict(x,50,m*500,U),'epochs',m*500/U)
 J=np.stack([((predict(x+np.eye(len(x))[i]*1e-5,N,T,P)-predict(x,N,T,P))/1e-5).ravel() for i in range(len(x))],axis=1)
 cov=np.linalg.inv(J.T@J)*.004**2
 rng=np.random.default_rng(3);samples=rng.multivariate_normal(x,cov,100)
 out=[optimize(xx) for xx in samples]
 print('mixture std',np.std([a for a,b in out],axis=0),'loss sd',np.std([b for a,b in out]))
 print('loss bounds',np.percentile([b for a,b in out],[1,50,99]))
