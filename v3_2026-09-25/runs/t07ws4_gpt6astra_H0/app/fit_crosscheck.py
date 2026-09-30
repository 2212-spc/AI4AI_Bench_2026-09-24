import numpy as np
from fit_scaling import rows,unpack,fit,model
n,d,f,s,y,lr=unpack(rows)
# Separate nuisance offsets for every observed schedule state; WSD completion
# is the reference. This tests the parametric LR-power assumption.
levels=sorted(set(np.round(lr,8))-{0.})
Z=np.array([np.isclose(lr,v,atol=1e-8).astype(float) for v in levels]).T

def solve(ab):
 a,b=ab
 X=np.c_[np.ones(len(y)),n**(-a),(d*f)**(-b),Z]
 p=np.linalg.lstsq(X,y,rcond=None)[0]
 return X@p-y,p
ab=np.array([.365,.345]);lam=.001
for i in range(100):
 res,p=solve(ab)
 J=np.array([(solve(ab+np.eye(2)[j]*1e-5)[0]-res)/1e-5 for j in range(2)]).T
 step=np.linalg.solve(J.T@J+lam*np.eye(2),-J.T@res)
 if np.sum(solve(ab+step)[0]**2)<sum(res**2):
  ab+=step;lam=max(lam/3,1e-10)
  if max(abs(step))<1e-9:break
 else:lam*=5
res,p=solve(ab)
a,b=ab
print('exponents',ab,'coefficients',p[:3],'rmse',np.mean(res**2)**.5)
print('offsets',dict(zip(levels,p[3:])))
def predict(N,D,f,s):
 rr=.55+.45*np.cos(np.pi*f) if s=='cosine' else min(1,(1-f)*5)
 offset=0 if rr==0 else p[3+int(np.argmin(abs(np.array(levels)-rr)))]
 return p[0]+p[1]*(N/1e8)**-a+p[2]*(D*f/1e9)**-b+offset
W=predict(3e9,6e10,1,'wsd');short=predict(3e9,2.4e10,1,'wsd');C=predict(3e9,6e10,1,'cosine')
print('q2',W,'q3',predict(3e9,6e10,.4,'cosine')-short,'q4',[predict(3e9,6e10,t,'cosine')-C for t in [.9,.5]],'q5 difference',predict(3e9,6e10,.7,'cosine')-short)
print('q1 losses',[predict(N,1.8e20/N,1,'cosine') for N in [1.2e9,2.4e9,4.8e9,9.6e9]])
