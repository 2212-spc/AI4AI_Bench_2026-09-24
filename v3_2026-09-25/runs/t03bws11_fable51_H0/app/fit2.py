import numpy as np
exec(open('/app/fit.py').read().split("# fresh data")[0])
p=np.load('/app/p_fresh.npy'); E,lA,a,lB,b=p; A=np.exp(lA); B=np.exp(lB)
def Lf(N,Deff): return E+A*N**-a+B*Deff**-b
def Deff_from(N,loss): return ((loss-E-A*N**-a)/B)**(-1/b)
Nn=5e7
print("== repetition (N=5e7, D=4e9)")
rep=[(2e11,4.4665),(2e9,4.4767),(1e9,4.5171),(4e8,4.636)]
for sub,l in rep:
    de=Deff_from(Nn,l); print(f"sub={sub:.1e} epochs={4e9/sub:.1f} Deff={de:.3e} Deff/U={de/min(sub,4e9):.3f} Deff/D={de/4e9:.3f}")
print("== filter fresh")
fil=[(4e9,0,4.4665),(4e9,0.3,4.4384),(4e9,0.6,4.3886),(4e9,0.85,4.3236),(1e9,0,4.9735),(1e9,0.6,4.8283)]
for D,q,l in fil:
    de=Deff_from(Nn,l); print(f"D={D:.0e} q={q} Deff/D={de/D:.3f}")
