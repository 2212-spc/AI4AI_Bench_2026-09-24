import numpy as np
a=0.1022
runs=[(np.log(0.0048),np.log(0.0048*0.055),3.00556),(np.log(0.0072),np.log(0.0072*0.0607),2.97532)]
sig=0.0017
def post(mu_u=np.log(0.0083),s_u=0.13,mu_p=-7.94,s_p=0.09,mu_L=2.974,s_L=0.009,mu_b=0.13,s_b=0.05,rec=None,verbose=True):
    U=np.linspace(mu_u-4*s_u,mu_u+4*s_u,81); P=np.linspace(mu_p-4*s_p,mu_p+4*s_p,61)
    L=np.linspace(mu_L-4*s_L,mu_L+4*s_L,61); Bv=np.linspace(max(0.03,mu_b-2.5*s_b),mu_b+2.5*s_b,25)
    u,p,l,b=np.meshgrid(U,P,L,Bv,indexing='ij')
    lp=-0.5*(((u-mu_u)/s_u)**2+((p-mu_p)/s_p)**2+((l-mu_L)/s_L)**2+((b-mu_b)/s_b)**2)
    for (ui,zi,yi) in runs: lp+=-0.5*((yi-(l+a*(ui-u)**2+b*(zi-p)**2))/sig)**2
    w=np.exp(lp-lp.max()); w/=w.sum()
    Eu=(w*u).sum(); Ep=(w*p).sum()
    if rec is None: rec=(Eu,Ep)
    val=l+a*(rec[0]-u)**2+b*(rec[1]-p)**2
    m=(w*val).sum(); s=np.sqrt((w*(val-m)**2).sum())
    pen=(w*(a*(rec[0]-u)**2+b*(rec[1]-p)**2)).sum()
    flat=val.ravel(); ww=w.ravel(); o=np.argsort(flat); c=np.cumsum(ww[o]); fs=flat[o]
    # best 0.011-wide interval
    best=(0,0)
    for lo in np.linspace(m-0.012,m+0.001,131):
        cov=c[np.searchsorted(fs,lo+0.011)-1]-c[np.searchsorted(fs,lo)] if np.searchsorted(fs,lo)>0 else c[np.searchsorted(fs,lo+0.011)-1]
        if cov>best[0]: best=(cov,lo)
    if verbose: print(f"lr={np.exp(Eu):.5f} wd={np.exp(Ep-Eu):.4f} | recipe loss {m:.4f}±{s:.4f}, E[pen]={pen:.5f}, P(pen>0.0006)={(w*((a*(rec[0]-u)**2+b*(rec[1]-p)**2)>0.0006)).sum():.2f} | best interval [{best[1]:.4f},{best[1]+0.011:.4f}] cov={best[0]:.2f}")
    return Eu,Ep,m,s
Eu,Ep,m,s=post()
print("--- sensitivity (recipe fixed at baseline posterior mean) ---")
for kw in [dict(s_u=0.2),dict(mu_u=np.log(0.0075)),dict(mu_u=np.log(0.0092)),dict(mu_p=-7.84),dict(mu_p=-8.04),dict(mu_L=2.966),dict(mu_L=2.982),dict(mu_b=0.09),dict(mu_b=0.18),dict(s_L=0.02)]:
    print(kw, end=' '); post(rec=(Eu,Ep),**kw)
