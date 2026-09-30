import numpy as np
exec(open('fit.py').read().split("p0=")[0])
def m1(p,lN,lD,lLR):
    E,A,al,Bc,be,k,b,c,plo,phi=p
    base=E+np.exp(A-al*lN)+np.exp(Bc-be*lD)
    x=lLR-(k+b*lN+c*lD); return base+np.where(x<0,plo,phi)*x**2
# curvature scaling with N and D
def m5(p,lN,lD,lLR):
    E,A,al,Bc,be,k,b,c,plo,phi,gN,gD=p
    base=E+np.exp(A-al*lN)+np.exp(Bc-be*lD)
    x=lLR-(k+b*lN+c*lD); sc=np.exp(gN*(lN-np.log(6e7))+gD*(lD-np.log(3e9)))
    return base+sc*np.where(x<0,plo,phi)*x**2
# optimum with quadratic term in lnD (check for curvature in D-dependence)
def m6(p,lN,lD,lLR):
    E,A,al,Bc,be,k,b,c,plo,phi,c2=p
    base=E+np.exp(A-al*lN)+np.exp(Bc-be*lD)
    x=lLR-(k+b*lN+c*lD+c2*(lD-np.log(3e9))**2); return base+np.where(x<0,plo,phi)*x**2
def fit(model,p0,mask=None):
    m=np.ones(len(L),bool) if mask is None else mask
    def res(p): return model(p,lN[m],lD[m],lLR[m])-L[m]
    p,r,cost=lm(res,p0,iters=400); return p,np.sqrt(cost/m.sum())
def ans(model,p):
    def lropt(N,D):
        g=np.linspace(np.log(1e-5),np.log(0.05),20001)
        v=model(p,np.log(N)*np.ones_like(g),np.log(D)*np.ones_like(g),g); return g[np.argmin(v)]/np.log(10),v.min()
    o1,_=lropt(1e9,2e10); o2,L2=lropt(1e9,1e12); o7,_=lropt(1e9,2e11)
    Lp=model(p,np.log(1e9),np.log(1e12),np.log(0.0007182))
    opts=[0.000165,0.00033,0.000661,0.00132]; l7=np.array([model(p,np.log(1e9),np.log(2e11),np.log(o)) for o in opts])
    return dict(q1=o1,q2=o2,q3=Lp-L2,q7=o7,dB_C=l7[1]-l7[2],q8=L2,b=p[6],c=p[7])
p0=[1.9,6.7,0.35,7.0,0.34,0.29,-0.11,-0.21,0.05,0.12]
p1,rms=fit(m1,p0); print('m1 rms %.4f'%rms, {k:round(float(v),4) for k,v in ans(m1,p1).items()})
p5,rms=fit(m5,p0+[0,0]); print('m5 rms %.4f'%rms,'gN=%.3f gD=%.3f'%(p5[10],p5[11]), {k:round(float(v),4) for k,v in ans(m5,p5).items()})
p6,rms=fit(m6,p0+[0]); print('m6 rms %.4f'%rms,'c2=%.4f'%p6[10], {k:round(float(v),4) for k,v in ans(m6,p6).items()})
# leave-one-sweep-out
keys=sorted(set(zip(N,D)))
print('\nleave-one-sweep-out (m1):')
for kk in keys:
    mask=~((N==kk[0])&(D==kk[1]))
    p,rms=fit(m1,p1,mask); a=ans(m1,p)
    print('drop N=%.1e D=%.1e'%kk, {k:round(float(v),4) for k,v in a.items()})
