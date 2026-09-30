import numpy as np
exec(open('/app/boot.py').read().split("# noise model")[0])
sig=(N/3e7)**(-0.35); w=1/sig
def pen_of(shape,t,u,lnn,lnd):
    if shape.startswith('asym'):
        pen=np.where(u<0,np.exp(t[3]),np.exp(t[4]))*u*u
    elif shape.startswith('pow'):
        kl,kr,pl,pr=np.exp(t[3]),np.exp(t[4]),t[5],t[6]; pen=np.where(u<0,kl*np.abs(u)**pl,kr*np.abs(u)**pr)
    else: k,s=np.exp(t[3]),t[4]; pen=k*(np.exp(s*u)-1-s*u)/s**2
    if shape.endswith('N'): pen=pen*np.exp(t[-2]*lnn+t[-1]*lnd)
    return pen
def resid(t,shape):
    u=lnlr-(t[0]+t[1]*lnN+t[2]*lnD); pen=pen_of(shape,t,u,lnN,lnD)
    y=L-pen; Lmin=np.array([np.sum((w**2*y)[G==i])/np.sum(w[G==i]**2) for i in range(ng)])
    return L-(Lmin[G]+pen),Lmin
X0={'asym':[-5.3,-0.15,-0.2,-2.9,-2.0],'asymN':[-5.3,-0.15,-0.2,-2.9,-2.0,0,0],'expq':[-5.4,-0.15,-0.2,-1.8,0.5],'expqN':[-5.4,-0.15,-0.2,-1.8,0.5,0,0],
    'pow':[-5.4,-0.15,-0.2,-2.5,-2.0,2.0,2.0],'powN':[-5.4,-0.15,-0.2,-2.5,-2.0,2.0,2.0,0,0]}
ST={k:[0.1,0.05,0.05]+[0.3]*(len(v)-3) for k,v in X0.items()}
for k in ('pow','powN'): ST[k][5]=ST[k][6]=0.2
res={}
for shape in X0:
    f=lambda t: np.sum((resid(t,shape)[0]*w)**2)
    t,fv=nm(f,X0[shape],ST[shape]); t,fv=nm(f,t,np.array(ST[shape])*0.3); t,fv=nm(f,t,np.array(ST[shape])*0.1)
    r,Lmin=resid(t,shape); n=len(L); k=len(t)+ng
    aic=n*np.log(fv/n)+2*k+2*k*(k+1)/(n-k-1)
    c,a,b=t[:3]; l1=(c+a*np.log(1e9/3e7)+b*np.log(2e10/3e9))/np.log(10); l2=(c+a*np.log(1e9/3e7)+b*np.log(1e12/3e9))/np.log(10)
    u=np.log(0.001453)-l2*np.log(10); pen=pen_of(shape,t,np.array([u]),np.log(1e9/3e7),np.log(1e12/3e9))[0]
    l7=(c+a*np.log(1e9/3e7)+b*np.log(2e11/3e9))/np.log(10)
    print('%-6s AICc=%7.1f wrms=%.4f  q1=%.3f q2=%.3f q3=%.4f q7lr*=%.5f  a=%.3f b=%.3f'%(shape,aic,np.sqrt(fv/(n-k)),l1,l2,pen,10**l7,a,b), np.round(t[3:],3))
    res[shape]=(t,Lmin,aic)
# dense group check
i=[j for j,k in enumerate(keys) if k==(1.2e8,2.4e9)][0]
m=G==i; o=np.argsort(lnlr[m])
print('dense group lr, loss:'); print(np.c_[np.exp(lnlr[m][o]),L[m][o]])
for shape in res:
    t=res[shape][0]; print(shape,'pred lr* at dense group %.5f'%np.exp(t[0]+t[1]*np.log(4)+t[2]*np.log(0.8)))
# local fits on dense group: points within factor ~2 of min
x=lnlr[m][o]; y=L[m][o]
for lo,hi in [(1,8),(2,8),(1,7),(2,7),(3,8)]:
    p=np.polyfit(x[lo:hi],y[lo:hi],2); print('parab pts',lo,hi,'lr*=%.5f'%np.exp(-p[1]/(2*p[0])))
    p=np.polyfit(x[lo:hi],y[lo:hi],3); rts=np.roots(np.polyder(p)); rts=rts[np.isreal(rts)].real; rts=rts[(rts>x[lo])&(rts<x[hi-1])]; print('   cubic lr*',np.exp(rts))
np.save('/app/final_res.npy',{'keys':keys,'res':{k:(v[0],v[1]) for k,v in res.items()}},allow_pickle=True)
