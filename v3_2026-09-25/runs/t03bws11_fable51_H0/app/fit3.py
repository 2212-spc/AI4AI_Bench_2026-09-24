import numpy as np
p=np.load('/app/p_fresh.npy'); E,lA,a,lB,b=p; A=np.exp(lA); B=np.exp(lB)
def Lf(N,Deff): return E+A*N**-a+B*Deff**-b
def Deff_from(N,loss): return ((loss-E-A*N**-a)/B)**(-1/b)
# repetition fit
rep=[(2e9,4.4767),(1e9,4.5171),(4e8,4.636)]
def Drep(D,U,R):
    e=D/U
    return np.where(e<=1, D, U*(1+R*(1-np.exp(-(e-1)/R))))
for R in [5,5.5,6,6.5,7,7.5,8]:
    pred=[Lf(5e7,Drep(4e9,s,R)) for s,_ in rep]
    print("R*",R,"resid",np.round(np.array(pred)-np.array([l for _,l in rep]),4))
# notebook ablation: N=5e7 D=8e9 sub=2.5e9
obs={0:4.3094,0.3:4.3168,0.6:4.3047}
R=6.5
print("== notebook ablation predictions")
# model 1: multiplier only, m from D=4e9 fresh
m1={0:1.0,0.3:1.099,0.6:1.307}
# model 2: multiplier + penalty; solve at q=0.6 from two D points; q=0.3 unknown -> skip
BD=lambda D: B*D**-b
for q in [0,0.3,0.6]:
    U=2.5e9*(1-q); dr=Drep(8e9,U,R)
    print(q,"epochs",8e9/U,"pred m-only",round(float(Lf(5e7,dr*m1[q])),4),"obs",obs[q], "Deff needed", f"{Deff_from(5e7,obs[q]):.3e}", "Drep",f"{dr:.3e}")
# model2 at q=0.6: m=1.70, pen=0.07
dr=Drep(8e9,1e9,R); print("q=0.6 model2 pred",round(float(Lf(5e7,dr*1.70)+0.07),4))
