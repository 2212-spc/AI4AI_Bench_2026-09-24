from fit import *
n,d,lr,y,groups,idx=load()
for shape in ['asym','cubic','quartic','soft','sym']:
 def pred(p):
  x=np.log(lr)-p[0]-p[1]*np.log(n/1e8)-p[2]*np.log(d/1e9)
  if shape=='asym':pen=np.where(x<0,p[3],p[4])*x*x
  if shape=='cubic':pen=p[3]*x*x+p[4]*x*x*x
  if shape=='quartic':pen=p[3]*x*x+p[4]*x*x*x+p[5]*x**4
  if shape=='soft':pen=p[3]*x*x+p[4]*np.maximum(x,0)**3
  if shape=='sym':pen=p[3]*x*x
  k=4 if shape=='sym' else 6 if shape=='quartic' else 5
  return p[k:][idx]+pen
 p0=[-5.35,-.14,-.19,.08]
 if shape!='sym':p0+=[.12 if shape=='asym' else .008]
 if shape=='quartic':p0+=[.001]
 k=len(p0)
 p,r=ls(lambda p:pred(p)-y,p0+[min(y[idx==i]) for i in range(len(groups))])
 print(shape,'parameters',p[:k],'rmse',np.sqrt(np.mean(r*r)))
 gn,gd=np.array(groups).T
 def basefit(b):return b[0]+b[1]*(gn/1e8)**(-b[3])+b[2]*(gd/1e9)**(-b[4])
 b,br=ls(lambda b:basefit(b)-p[k:],[1.5,1,2,.3,.3])
 print('base',b,'residuals',br,'prod',b[0]+b[1]*10**(-b[3])+b[2]*1000**(-b[4]))
 print('targets',[(D,np.exp(p[0]+p[1]*np.log(10)+p[2]*np.log(D/1e9))) for D in [2e10,2e11,1e12]])
