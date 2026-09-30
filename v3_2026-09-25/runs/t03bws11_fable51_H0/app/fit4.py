import numpy as np
exec(open('/app/fit.py').read().split("# fresh data")[0])
# all data: (N,D,q,sub,loss)
data=[(1.5e7,3e8,0,2e11,6.3366),(3e7,6e8,0,2e11,5.4907),(6e7,1.2e9,0,2e11,4.8085),(1.2e8,2.4e9,0,2e11,4.2527),(2.4e8,4.8e9,0,2e11,3.8118),
(5e7,8e9,0,2.5e9,4.3098),(5e7,8e9,0,2.5e9,4.309),(5e7,8e9,0.3,2.5e9,4.316),(5e7,8e9,0.3,2.5e9,4.3176),(5e7,8e9,0.6,2.5e9,4.3065),(5e7,8e9,0.6,2.5e9,4.3029),
(5e7,1e9,0,2e11,4.9735),(5e7,4e9,0,2e11,4.4665),(5e7,1.6e10,0,2e11,4.1176),(5e7,4e9,0,4e8,4.636),(5e7,4e9,0,1e9,4.5171),(5e7,4e9,0,2e9,4.4767),
(5e7,4e9,0.3,2e11,4.4384),(5e7,4e9,0.6,2e11,4.3886),(5e7,4e9,0.85,2e11,4.3236),(5e7,1e9,0.6,2e11,4.8283),
(5e7,1.6e10,0.6,2e11,4.0831),(5e7,4e9,0.6,1e9,4.5425)]
N=np.array([d[0] for d in data]);D=np.array([d[1] for d in data]);Q=np.array([d[2] for d in data]);S=np.array([d[3] for d in data]);L=np.array([d[4] for d in data])
def Drep(D,U,R):
    e=D/U
    return np.where(e<=1, D, U*(1+R*(1-np.exp(-np.maximum(e-1,0)/R))))
qlev=[0.3,0.6,0.85]
def unpack(p):
    E,lA,a,lB,b,lR=p[:6]; v={0:1.0}; pen={0:0.0}
    for i,q in enumerate(qlev): v[q]=1+np.exp(p[6+2*i]); pen[q]=p[7+2*i]
    return E,np.exp(lA),a,np.exp(lB),b,np.exp(lR),v,pen
def pred(p,N,D,Q,S):
    E,A,a,B,b,R,v,pen=unpack(p)
    out=[]
    for n,d,q,s in zip(N,D,Q,S):
        U=s*(1-q); de=Drep(d,U,R)*v[q]
        out.append(E+A*n**-a+B*de**-b+pen[q])
    return np.array(out)
def obj(p): return np.sum((pred(p,N,D,Q,S)-L)**2)
pf=np.load('/app/p_fresh.npy')
x0=list(pf)+[np.log(6.5)]+[np.log(0.1),0.0,np.log(0.7),0.07,np.log(1.5),0.1]
p,val=nm(obj,x0,iters=30000,step=0.2)
for k in range(4): p,val=nm(obj,p,iters=30000,step=0.05)
print("rss",val,"rms",np.sqrt(val/len(L)))
E,A,a,B,b,R,v,pen=unpack(p)
print("E,A,a,B,b,R",E,A,a,B,b,R); print("v",v); print("pen",pen)
r=pred(p,N,D,Q,S)-L
for d,rr in zip(data,r): print(d, round(float(rr),4))
np.save('/app/p_joint.npy',p)
