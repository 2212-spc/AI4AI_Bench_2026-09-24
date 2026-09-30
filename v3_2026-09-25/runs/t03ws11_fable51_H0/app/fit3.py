import numpy as np
from fit import *
fresh=(q==0)&(D<=sub)
def mf(p,N,D):
    E,A,al,B,be=p; return E+A*N**(-al)+B*D**(-be)
fun=lambda p: mf(p,N[fresh],D[fresh])-L[fresh]
best=None
for E0 in [0.5,1.5,2.5]:
  for al0 in [0.2,0.4]:
    for be0 in [0.15,0.3]:
      p,r=lm(fun,[E0,50,al0,50,be0],400)
      if best is None or r@r<best[1]@best[1]: best=(p,r)
pf,r=best
print('fresh fit E,A,al,B,be',np.round(pf,4),'rms',np.sqrt(np.mean(r**2)), 'n',fresh.sum())
E,A,al,B,be=pf
# seed noise at N=5e7 D=4e9 q=0: 
for qq in [0,0.5,0.6]:
    m=(N==5e7)&(D==4e9)&(q==qq)&(sub==2e11); print('q',qq,'losses',L[m],'mean',L[m].mean(),'sd',L[m].std(ddof=1))
# implied Deff
U=sub*(1-q); T=L-E-A*N**(-al); De=(T/B)**(-1/be)
print('\nfilter multipliers (fresh, q>0):')
for i in np.where((q>0)&(D<=U))[0]:
    print(f"N={N[i]:.0e} D={D[i]:.0e} q={q[i]} s={int(seed[i])} L={L[i]:.4f} dL={L[i]-mf(pf,N[i],D[i]):+.4f} mult={De[i]/D[i]:.3f}")
print('\nrepetition (q=0):')
for i in np.where((q==0)&(D>U))[0]:
    print(f"N={N[i]:.0e} D={D[i]:.0e} U={U[i]:.2e} ep={D[i]/U[i]:.2f} L={L[i]:.4f} Deff/U={De[i]/U[i]:.3f}")
print('\nfilter+repetition:')
for i in np.where((q>0)&(D>U))[0]:
    print(f"N={N[i]:.0e} D={D[i]:.0e} q={q[i]} U={U[i]:.2e} ep={D[i]/U[i]:.2f} L={L[i]:.4f} Deff/U={De[i]/U[i]:.3f}")
np.save('pf.npy',pf)
