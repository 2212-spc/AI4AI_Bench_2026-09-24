import numpy as np
exec(open('fit.py').read().split("p0=")[0])
def run(model,p0,label):
    def res(p): return model(p,lN,lD,lLR)-L
    p,r,cost=lm(res,p0,iters=400)
    print('\n==',label,'rms=%.4f'%np.sqrt(cost/len(r)))
    return p
# model 1: asymmetric quadratic in ln lr, power-law optimum
def m1(p,lN,lD,lLR):
    E,A,al,Bc,be,k,b,c,plo,phi=p
    base=E+np.exp(A-al*lN)+np.exp(Bc-be*lD)
    x=lLR-(k+b*lN+c*lD); return base+np.where(x<0,plo,phi)*x**2
# model 2: penalty quadratic + cubic (smooth asymmetry)
def m2(p,lN,lD,lLR):
    E,A,al,Bc,be,k,b,c,q,cu=p
    base=E+np.exp(A-al*lN)+np.exp(Bc-be*lD)
    x=lLR-(k+b*lN+c*lD); return base+q*x**2+cu*x**3
# model 3: multiplicative penalty on reducible loss
def m3(p,lN,lD,lLR):
    E,A,al,Bc,be,k,b,c,plo,phi=p
    red=np.exp(A-al*lN)+np.exp(Bc-be*lD)
    x=lLR-(k+b*lN+c*lD); return E+red*(1+np.where(x<0,plo,phi)*x**2)
# model 4: penalty  a*(exp(x)-1-x)  style (exp-asymmetric)
def m4(p,lN,lD,lLR):
    E,A,al,Bc,be,k,b,c,q,s=p
    base=E+np.exp(A-al*lN)+np.exp(Bc-be*lD)
    x=lLR-(k+b*lN+c*lD); return base+q*(np.exp(s*x)-1-s*x)/s**2
p1=run(m1,[1.9,6.7,0.35,7.0,0.34,0.29,-0.11,-0.21,0.05,0.12],'asym quad')
p2=run(m2,[1.9,6.7,0.35,7.0,0.34,0.29,-0.11,-0.21,0.08,0.02],'quad+cubic')
p3=run(m3,[1.9,6.7,0.35,7.0,0.34,0.29,-0.11,-0.21,0.03,0.06],'multiplicative')
p4=run(m4,[1.9,6.7,0.35,7.0,0.34,0.29,-0.11,-0.21,0.08,1.0],'exp-asym')
names='E lnA al lnB be k b c x1 x2'.split()
for lab,p in [('m1',p1),('m2',p2),('m3',p3),('m4',p4)]: print(lab,{n:round(v,4) for n,v in zip(names,p)})

def answers(model,p,lab):
    E,A,al,Bc,be,k,b,c=p[:8]
    def lropt(N,D):
        # numerically minimise model in lr
        g=np.linspace(np.log(1e-5),np.log(0.05),20001)
        v=model(p,np.log(N)*np.ones_like(g),np.log(D)*np.ones_like(g),g); return np.exp(g[np.argmin(v)]),v.min()
    o1,_=lropt(1e9,2e10); o2,L2=lropt(1e9,1e12); o7,_=lropt(1e9,2e11)
    Lprod=model(p,np.log(1e9),np.log(1e12),np.log(0.0007182))
    opts=[0.000165,0.00033,0.000661,0.00132]
    l7=[model(p,np.log(1e9),np.log(2e11),np.log(o)) for o in opts]
    print(lab,'q1 %.4f q2 %.4f q3 %.4f q4[%.4f,%.4f] q5 10x N ratio %.3f  q7 %s %s q8 %.4f'%(
      np.log10(o1),np.log10(o2),Lprod-L2,np.log10(o2),np.log10(o2)+0.5*np.log10(8),10**b,
      'ABCD'[int(np.argmin(l7))],np.round(np.array(l7)-min(l7),4),L2), ' lr*7=%.6f'%o7, 'q6 lr .00173 log=%.4f'%np.log10(0.00173))
for lab,m,p in [('m1',m1,p1),('m2',m2,p2),('m3',m3,p3),('m4',m4,p4)]: answers(m,p,lab)
np.save('p1.npy',p1)
