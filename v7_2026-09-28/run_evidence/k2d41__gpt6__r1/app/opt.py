from fit import *
W=np.array([.095,.285,.62])
def objective(p,w):
 w=np.atleast_2d(w);m=len(w)
 return predict(p,(np.full(m,50.),np.full(m,500.),w,np.tile(U/1e9,(m,1))))@W

def opt(p):
 rng=np.random.default_rng(5);ws=rng.dirichlet([2,2,2,2],10000);w=ws[np.argmin(objective(p,ws))].copy()
 for it in range(30):
  old=w.copy()
  for a in range(4):
   for b in range(a+1,4):
    s=w[a]+w[b];lo=0;hi=s
    for k in range(30):
     l=lo+(hi-lo)*.381966;r=lo+(hi-lo)*.618034
     ww=np.tile(w,(2,1));ww[:,a]=[l,r];ww[:,b]=s-ww[:,a]
     v=objective(p,ww)
     if v[0]<v[1]:hi=r
     else:lo=l
    w[a]=(lo+hi)/2;w[b]=s-w[a]
  if max(abs(w-old))<1e-7:break
 return w,objective(p,w)[0]
if __name__=='__main__':
 p=np.load('/app/model.npy');w,v=opt(p);print(w,v);np.save('/app/optimum.npy',w)
