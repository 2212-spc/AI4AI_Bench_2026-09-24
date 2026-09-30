from fit import *
n,d,lr,y,groups,idx=load()
# Separate size and horizon power laws for baseline loss, and a power-law
# optimum with different curvature on the two sides in log learning rate.
def pred(p):
 x=np.log(lr)-p[0]-p[1]*np.log(n/1e8)-p[2]*np.log(d/1e9)
 base=p[5]+p[6]*(n/1e8)**(-p[8])+p[7]*(d/1e9)**(-p[9])
 return base+np.where(x<0,p[3],p[4])*x*x
for wexp in [0,.15,.3,.5]:
 w=(n/1e8)**wexp
 p,r=ls(lambda p:(pred(p)-y)*w,[-5.35,-.14,-.19,.063,.115,1.7,1.23,1.67,.326,.373])
 rawr=r/w
 print('weight exponent',wexp,'params',p,'RMSE',np.sqrt(np.mean(rawr**2)))
 for N in sorted(set(n)):print('N',N,'RMS',np.sqrt(np.mean(rawr[n==N]**2)))
 target=np.array([p[0]+p[1]*np.log(10)+p[2]*np.log(D/1e9) for D in [2e10,2e11,1e12]])
 q3=p[4]*(np.log(.001453)-target[2])**2
 q8=p[5]+p[6]*10**-p[8]+p[7]*1000**-p[9]
 print('target log10 lrs',target/np.log(10),'q3',q3,'q8',q8)
