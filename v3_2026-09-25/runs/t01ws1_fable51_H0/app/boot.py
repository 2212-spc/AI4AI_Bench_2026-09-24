import numpy as np
exec(open('/app/fit.py').read().split("th0=")[0])
th_hat=np.array([-6.3947,-0.1115,-0.2229,0.4436])
A=design(th_hat,'powerlaw'); beta=np.linalg.lstsq(A,ys,rcond=None)[0]; fit=A@beta; res=ys-fit
rng=np.random.default_rng(0); out=[]
ys_orig=ys.copy()
for bsi in range(60):
    ys=fit+rng.normal(0,0.0075,len(ys))   # seed-noise level from seed repeats / residuals
    th,_=nelder_mead(lambda t: sse(t), th_hat, step=[0.05,0.03,0.03,0.2], iters=1500)
    Ab=design(th,'powerlaw'); bb=np.linalg.lstsq(Ab,ys,rcond=None)[0]
    out.append(np.r_[th,bb[K]])
ys=ys_orig
out=np.array(out)
print("param means",out.mean(0).round(4)); print("param sds  ",out.std(0).round(4))
def lnlr(th,N,D): return th[0]+th[1]*np.log(N/1e8)+th[2]*np.log(D/2e9)
for N,D in [(1e9,2e10),(1e9,2e11),(1e9,1e12)]:
    v=np.array([lnlr(t,N,D) for t in out])/np.log(10)
    print("N=%.0e D=%.0e log10 lr* = %.3f  boot sd %.3f"%(N,D,lnlr(th_hat,N,D)/np.log(10),v.std()))
# q3 excess
def excess(t,k,N,D,lr):
    u=np.log(lr)-lnlr(t,N,D); return k*shape(u,t[3])
e=np.array([excess(t[:4],t[4],1e9,1e12,0.0007182) for t in out])
print("q3 excess: point %.4f boot sd %.4f"%(excess(th_hat,beta[K],1e9,1e12,0.0007182),e.std()))
# q5: 10x N at fixed D: factor
f=np.exp(out[:,1]*np.log(10)); print("q5 factor 10^b: point %.3f  boot range %.3f..%.3f"%(10**th_hat[1],f.min(),f.max()))
# q7
for lr in [0.000165,0.00033,0.000661,0.00132]:
    print("q7 lr=%.6f excess=%.4f"%(lr,excess(th_hat,beta[K],1e9,2e11,lr)))
