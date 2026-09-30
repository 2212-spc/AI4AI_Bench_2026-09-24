import numpy as np
exec(open('/app/fit.py').read().split("th0=")[0])
# Model B: free lr* per sweep and free k per sweep, shared s
def sseB(theta):
    s=theta[0]; ustar=theta[1:]
    u=llr-ustar[idx]
    A=np.zeros((len(ys),2*K)); A[np.arange(len(ys)),idx]=1
    A[np.arange(len(ys)),K+idx]=shape(u,s)
    beta=np.linalg.lstsq(A,ys,rcond=None)[0]; r=ys-A@beta; return (r*r).sum(),beta
th0=np.r_[0.44,[np.log(0.0017)-0.11*np.log(k[0]/1e8)-0.22*np.log(k[1]/2e9) for k in keys]]
thB,vB=nelder_mead(lambda t: sseB(t)[0], th0, step=0.1, iters=20000)
_,betaB=sseB(thB)
print("model B: s=%.3f sse=%.5f  (model A sse=%.5f)"%(thB[0],vB,sse([-6.3947,-0.1115,-0.2229,0.4436])))
print("sweep     N        D     ln lr*_free  ln lr*_PL   diff   k_free")
for i,k in enumerate(keys):
    pl=-6.3947-0.1115*np.log(k[0]/1e8)-0.2229*np.log(k[1]/2e9)
    print("%2d %.1e %.1e  %7.3f  %7.3f  %6.3f  %.3f"%(i,k[0],k[1],thB[1+i],pl,thB[1+i]-pl,betaB[K+i]))
# regress k_free on lnN, lnD
X=np.c_[np.ones(K),[np.log(k[0]/1e8) for k in keys],[np.log(k[1]/2e9) for k in keys]]
cf=np.linalg.lstsq(X,betaB[K:],rcond=None)[0]; print("k ~ %.3f + %.3f lnN + %.3f lnD"%tuple(cf))
cf2=np.linalg.lstsq(X,thB[1:],rcond=None)[0]; print("free ln lr* ~ %.3f + %.3f lnN + %.3f lnD"%tuple(cf2))
