import numpy as np
from fit import *
pf=np.load('pf.npy'); E,A,al,B,be=pf
U=sub*(1-q); ep=D/U; R=np.maximum(ep-1,0)
def f_rep(Rs): return np.where(D<=U,ep,1+Rs*(1-np.exp(-R/Rs)))
# Model M1: Deff = m(q)*U*f ;  m=exp(m1 q + m2 q^2)
def M1(p):
    E,A,al,B,be,Rs,m1,m2=p
    De=U*f_rep(Rs)*np.exp(m1*q+m2*q*q)
    return E+A*N**(-al)+B*De**(-be)
# Model M2: additive gain: L = base(Deff_rep) - c(q) * Deff^-g, c=c1 q + c2 q^2
def M2(p):
    E,A,al,B,be,Rs,c1,c2,g=p
    De=U*f_rep(Rs)
    return E+A*N**(-al)+B*De**(-be)-(c1*q+c2*q*q)*(De/1e9)**(-g)
# Model M3: multiplier decays with D: m = 1 + (k1 q + k2 q^2)*(De/1e9)^-g
def M3(p):
    E,A,al,B,be,Rs,k1,k2,g=p
    De=U*f_rep(Rs)
    m=1+(k1*q+k2*q*q)*(De/1e9)**(-g)
    return E+A*N**(-al)+B*(De*m)**(-be)
# Model M5: multiplier + constant penalty p(q)= p1 q + p2 q^2
def M5(p):
    E,A,al,B,be,Rs,m1,m2,p1,p2=p
    De=U*f_rep(Rs)*np.exp(m1*q+m2*q*q)
    return E+A*N**(-al)+B*De**(-be)+p1*q+p2*q*q
for name,M,p0 in [('M1',M1,list(pf)+[6.5,0.5,0.0]),('M2',M2,list(pf)+[6.5,0.1,0.1,0.5]),('M3',M3,list(pf)+[6.5,0.5,0.0,0.3]),('M5',M5,list(pf)+[6.5,0.8,0,0.05,0])]:
    fun=lambda p: M(p)-L
    p,r=lm(fun,p0,600)
    print(name,'rms',round(float(np.sqrt(np.mean(r**2))),5),'params',np.round(p,4))
    np.save(name+'.npy',p)
    # residuals for q>0 fresh, sorted by D
    idx=np.where(q>0)[0]; idx=idx[np.argsort(D[idx])]
    print('  q>0 resid:',' '.join(f"{D[i]:.0e}/q{q[i]}/{'F' if D[i]<=U[i] else 'R'}:{r[i]:+.3f}" for i in idx))
    idx=np.where((q==0)&(D>U))[0]
    print('  rep resid:',' '.join(f"ep{ep[i]:.0f}:{r[i]:+.3f}" for i in idx))
