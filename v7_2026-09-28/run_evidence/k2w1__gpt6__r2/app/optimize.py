from fullfit import *
W=np.array([.34,.33,.33])
def objective(p,m):return predict(p,np.array([25.]),np.array([m])*500,U[None,:])[0]@W
def optimize(p):
 m=np.array([.4,.35,.1,.15])
 for it in range(100):
  old=m.copy()
  for i in range(4):
   for j in range(i+1,4):
    s=m[i]+m[j];a,b=0.,s
    def f(x):
     mm=m.copy();mm[i]=x;mm[j]=s-x;return objective(p,mm)
    for k in range(45):
     c=a+(b-a)*.38196601125;d=b-(b-a)*.38196601125
     if f(c)<f(d):b=d
     else:a=c
    m[i]=(a+b)/2;m[j]=s-m[i]
  if max(abs(old-m))<1e-8:break
 return m,objective(p,m)
if __name__=='__main__':
 p=np.load('full_params.npy');m,l=optimize(p);print('OPTIMUM',m,l,'epochs',m*500/U)
 for mix in [[.25]*4,[.7,.1,.1,.1],[.4,.35,.1,.15]]:print(mix,objective(p,mix))
 rng=np.random.default_rng(731);rs=load();n,t,u,y=arrays(rs);yh=predict(p,n,t,u);samples=[]
 for i in range(100):
  yy=yh+rng.normal(0,.004,y.shape)
  pp,v=lm(lambda pp:(predict(pp,n,t,u)-yy).ravel(),p)
  samples.append(objective(pp,m))
 print('target uncertainty',np.percentile(samples,[0,2.5,50,97.5,100]),np.std(samples))
