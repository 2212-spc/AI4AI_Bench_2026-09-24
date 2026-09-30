import numpy as np
a=0.1022
runs=[(np.log(0.0048),np.log(0.0048*0.055),3.00556),(np.log(0.0072),np.log(0.0072*0.0607),2.97532)]
sig=0.0017
# priors
mu_u,s_u=np.log(0.0083),0.13
mu_p,s_p=-7.94,0.09
mu_L,s_L=2.974,0.009
mu_b,s_b=0.13,0.05
U=np.linspace(mu_u-4*s_u,mu_u+4*s_u,81)
P=np.linspace(mu_p-4*s_p,mu_p+4*s_p,61)
L=np.linspace(mu_L-4*s_L,mu_L+4*s_L,61)
Bv=np.linspace(0.04,0.28,25)
u,p,l,b=np.meshgrid(U,P,L,Bv,indexing='ij')
logpost=-0.5*(((u-mu_u)/s_u)**2+((p-mu_p)/s_p)**2+((l-mu_L)/s_L)**2+((b-mu_b)/s_b)**2)
for (ui,zi,yi) in runs:
    pred=l+a*(ui-u)**2+b*(zi-p)**2
    logpost+=-0.5*((yi-pred)/sig)**2
w=np.exp(logpost-logpost.max()); w/=w.sum()
Eu=(w*u).sum(); Ep=(w*p).sum(); EL=(w*l).sum(); Eb=(w*b).sum()
Vu=(w*(u-Eu)**2).sum(); Vp=(w*(p-Ep)**2).sum(); VL=(w*(l-EL)**2).sum()
print(f"post u*: {Eu:.3f}±{np.sqrt(Vu):.3f} -> lr*={np.exp(Eu):.5f}; p: {Ep:.3f}±{np.sqrt(Vp):.3f} -> lr*wd*={np.exp(Ep):.3e}, wd*={np.exp(Ep-Eu):.4f}; L={EL:.4f}±{np.sqrt(VL):.4f}; b={Eb:.3f}")
# expected penalty at chosen point (u=Eu, z=Ep)
pen=(w*(a*(Eu-u)**2+b*(Ep-p)**2)).sum()
print(f"expected penalty at posterior mean: {pen:.5f}; expected loss of recipe: {EL+pen:.4f}")
# distribution of the recipe's expected loss: L + a(Eu-u)^2 + b(Ep-p)^2
val=l+a*(Eu-u)**2+b*(Ep-p)**2
m=(w*val).sum(); s=np.sqrt((w*(val-m)**2).sum())
print(f"recipe expected loss: {m:.4f} ± {s:.4f}")
# quantiles
flat=val.ravel(); ww=w.ravel(); o=np.argsort(flat); c=np.cumsum(ww[o])
for q in [0.01,0.05,0.5,0.95,0.99]: print(q, flat[o][np.searchsorted(c,q)])
