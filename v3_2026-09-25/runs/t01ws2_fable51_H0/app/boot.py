import numpy as np, sys
exec(open('/app/joint.py').read().split("shapes={")[0])
def fit(shape,x0,st,w):
    def f(th):
        r,_=model_resid_w(th,shape,w); return np.sum((r*w)**2)
    th,fv=nm(f,x0,st); th,fv=nm(f,th,np.array(st)*0.3); return th,fv
def model_resid_w(theta,shape,w):
    c,a,b=theta[:3]; u=lnlr-(c+a*lnN+b*lnD)
    if shape=='asym': kl,kr=np.exp(theta[3]),np.exp(theta[4]); pen=np.where(u<0,kl,kr)*u**2
    elif shape=='asymN': kl,kr=np.exp(theta[3]),np.exp(theta[4]); p,q=theta[5],theta[6]; pen=np.exp(p*lnN+q*lnD)*np.where(u<0,kl,kr)*u**2
    elif shape=='expq': k,s=np.exp(theta[3]),theta[4]; pen=k*(np.exp(s*u)-1-s*u)/s**2
    elif shape=='expqN': k,s=np.exp(theta[3]),theta[4]; p,q=theta[5],theta[6]; pen=np.exp(p*lnN+q*lnD)*k*(np.exp(s*u)-1-s*u)/s**2
    y=L-pen; Lmin=np.array([np.sum((w**2*y)[G==i])/np.sum(w[G==i]**2) for i in range(ng)])
    return L-(Lmin[G]+pen), Lmin
# noise model: sigma ~ N^-gamma; estimate from unweighted residuals
x0={'asym':[np.log(0.003),-0.15,-0.2,-2.9,-2.0],'asymN':[np.log(0.003),-0.15,-0.2,-2.9,-2.0,0,0],'expq':[np.log(0.003),-0.15,-0.2,-1.8,0.5],'expqN':[np.log(0.003),-0.15,-0.2,-1.8,0.5,0,0]}
st={'asym':[0.1,0.05,0.05,0.3,0.3],'asymN':[0.1,0.05,0.05,0.3,0.3,0.1,0.1],'expq':[0.1,0.05,0.05,0.3,0.2],'expqN':[0.1,0.05,0.05,0.3,0.2,0.1,0.1]}
w0=np.ones(len(L))
th,_=fit('asym',x0['asym'],st['asym'],w0); r,_=model_resid_w(th,'asym',w0)
# sigma per N bucket
for n in sorted(set(N)): print('N=%.1e rms resid %.4f n=%d'%(n,np.sqrt(np.mean(r[N==n]**2)),np.sum(N==n)))
gam=np.polyfit(np.log(N),np.log(np.abs(r)+1e-4),1)[0]; print('gamma approx',gam)
sig=(N/3e7)**(-0.35); w=1/sig
rng=np.random.default_rng(0)
for shape in ['asym','asymN','expq','expqN']:
    th,fv=fit(shape,x0[shape],st[shape],w); r,Lmin=model_resid_w(th,shape,w)
    print(shape,'weighted params',np.round(th,3),'chi2/dof %.4f'%(fv/(len(L)-len(th)-ng)))
    # parametric bootstrap
    Lsave=L.copy(); TH=[]
    for bnum in range(80):
        L[:]=Lsave-r+rng.standard_normal(len(L))*sig*np.sqrt(fv/(len(L)-len(th)-ng))
        tb,_=fit(shape,th,np.array(st[shape])*0.3,w); TH.append(tb)
    L[:]=Lsave; TH=np.array(TH)
    print('   boot sd',np.round(TH.std(0),3))
    # derived: log10 lr*(1e9,2e10), (1e9,1e12), penalty at lr_prod
    def derived(t):
        c,a,b=t[:3]; l1=c+a*np.log(1e9/3e7)+b*np.log(2e10/3e9); l2=c+a*np.log(1e9/3e7)+b*np.log(1e12/3e9)
        u=np.log(0.001453)-l2
        if shape.startswith('asym'):
            k=np.exp(t[3]) if u<0 else np.exp(t[4]); pen=k*u*u
        else: k,s=np.exp(t[3]),t[4]; pen=k*(np.exp(s*u)-1-s*u)/s**2
        if shape.endswith('N'): pen*=np.exp(t[5]*np.log(1e9/3e7)+t[6]*np.log(1e12/3e9))
        return np.array([l1/np.log(10),l2/np.log(10),pen,1-10**(a*1)])
    d0=derived(th); dB=np.array([derived(t) for t in TH])
    print('   q1 log10lr*=%.3f±%.3f  q2=%.3f±%.3f  q3 pen=%.4f±%.4f  q5 drop=%.3f±%.3f'%(d0[0],dB[:,0].std(),d0[1],dB[:,1].std(),d0[2],dB[:,2].std(),d0[3],dB[:,3].std()))
