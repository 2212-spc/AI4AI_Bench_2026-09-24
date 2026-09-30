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
p,r=best
print('fresh fit E,A,al,B,be',np.round(p,4),'rms',np.sqrt(np.mean(r**2)))
for i in np.where(fresh)[0]:
    print(f"N={N[i]:.1e} D={D[i]:.1e} L={L[i]:.4f} pred={mf(p,N[i],D[i]):.4f}")
np.save('pf.npy',p)
# implied Deff for each non-fresh run given N: solve B Deff^-be = L - E - A N^-al
E,A,al,B,be=p
print('\nnon-fresh runs: implied Deff/U and epochs')
for i in np.where(~fresh)[0]:
    U=sub[i]*(1-q[i]); T=L[i]-E-A*N[i]**(-al); De=(T/B)**(-1/be)
    print(f"N={N[i]:.1e} D={D[i]:.1e} q={q[i]} sub={sub[i]:.2e} U={U:.2e} ep={D[i]/U:.2f} L={L[i]:.4f} Deff={De:.3e} Deff/U={De/U:.3f} Deff/D={De/D[i]:.3f}")
