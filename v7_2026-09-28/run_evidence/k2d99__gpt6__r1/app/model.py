from fit import *

def effective(T,U,r):
 z=np.maximum(T/U-1,0)
 return np.minimum(T,U)+U*r*(-np.expm1(-z/r))

def prediction(x,N,T,U):
 E=effective(T,U,np.exp(x[24:28]))
 return np.array([pred(x[e*8:(e+1)*8],N,E) for e in range(3)]).T

def train():
 N,T,Y,U=load()
 x=np.r_[np.load('/app/basefit.npy').flatten(),np.log([15,2,1,5])]
 x[8:16]=[.5,np.log(.6),np.log(.35),np.log(.48),-2.4,.17,-3.37,-1.08]
 x,s=leastsq(lambda x:(prediction(x,N,T,U)-Y).flatten(),x,300)
 np.save('/app/model.npy',x)
 print('SSE',s,'rmse',np.sqrt(s/Y.size),'repetition',np.exp(x[24:]))
 for e in range(3): print(e,'c',x[e*8],'a',np.exp(x[e*8+1]),'alpha beta',np.exp(x[e*8+2:e*8+4]),'transfer',np.exp(x[e*8+4:e*8+8]))
 print('residuals',np.round(prediction(x,N,T,U)-Y,4))
 return x

def optimize(x):
 U=np.array([[3000,22.468979443588654,2.10893474102503,56.507653616707794]])
 def fun(m): return prediction(x,np.array([50]),np.array([m])*500,U)[0]@np.array([.249,.369,.382])
 rng=np.random.default_rng(10); M=rng.dirichlet([1]*4,20000)
 yy=prediction(x,np.ones(len(M))*50,M*500,U)@np.array([.249,.369,.382]); m=M[np.argmin(yy)]; best=fun(m)
 step=.05
 while step>1e-8:
  found=False
  for i in range(4):
   for j in range(4):
    if i==j:continue
    mm=m.copy();v=min(step,mm[i]);mm[i]-=v;mm[j]+=v
    val=fun(mm)
    if val<best: m=mm;best=val;found=True
  if not found:step*=.5
 return m,best
if __name__=='__main__':
 x=train();print('TARGET',optimize(x))
