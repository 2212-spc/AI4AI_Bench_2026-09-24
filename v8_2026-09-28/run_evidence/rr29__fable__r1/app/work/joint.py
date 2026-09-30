import json, numpy as np
R=[json.loads(l) for l in open('results.jsonl')]
groups=sorted(set((r['batch'],r['tokens_B']) for r in R)); gi={g:i for i,g in enumerate(groups)}; n=len(groups)
lr=np.array([r['lr'] for r in R]); wd=np.array([r['wd'] for r in R]); y=np.array([r['final_val_loss'] for r in R])
B=np.array([r['batch'] for r in R]); T=np.array([r['tokens_B'] for r in R]); steps=T*1e9/(B*4096)
g=np.array([gi[(r['batch'],r['tokens_B'])] for r in R])
def model(p, mode):
    L0=p[:n]; a=p[n]; c0,c1,c2=p[n+1:n+4]; q0,q1=p[n+4:n+6]; w0,w1=p[n+6:n+8]
    if mode=='const': q1=0.0
    lnlrs=c0+c1*np.log(steps)+c2*np.log(B/256)
    lnP=np.log(lr*wd*steps); aw=np.exp(w0)*T**w1
    return L0[g]+a*(np.log(lr)-lnlrs)**2+aw*(lnP-(q0+q1*np.log(steps)))**2
def jac(f,p,eps=1e-6):
    J=np.zeros((len(y),len(p)))
    for i in range(len(p)):
        d=np.zeros(len(p)); d[i]=eps; J[:,i]=(f(p+d)-f(p-d))/(2*eps)
    return J
def fit(mode):
    p=np.concatenate([[y[g==i].min() for i in range(n)],[0.11,-3.8,-0.31,0.44,np.log(3.7),0.0,np.log(0.02)-np.log(96)*1.2,1.2]])
    free=np.ones(len(p),bool)
    if mode=='const': free[n+5]=False
    lam=1e-3
    for it in range(200):
        f=lambda q: model(q,mode)
        r=f(p)-y; J=jac(f,p)[:,free]
        H=J.T@J+lam*np.diag(np.diag(J.T@J)); step=np.linalg.solve(H,-J.T@r)
        pn=p.copy(); pn[free]+=step
        if np.sum((f(pn)-y)**2)<np.sum(r**2): p=pn; lam*=0.3
        else: lam*=10
    r=model(p,mode)-y; J=jac(lambda q: model(q,mode),p)[:,free]
    dof=len(y)-free.sum(); s2=np.sum(r**2)/dof
    cov=np.linalg.inv(J.T@J)*s2
    return p,r,np.sqrt(np.diag(cov)),free
for mode in ['const','power']:
    p,r,sd,free=fit(mode); rms=np.sqrt(np.mean(r**2))
    print(f"== {mode}: rms resid {rms:.5f} (noise 0.0016), max |r| {np.abs(r).max():.4f}")
    names=['a','c0','c1','c2','q0','q1','w0','w1']; sdf=iter(sd)
    for i,nm in enumerate(names):
        idx=n+i; print(f"  {nm}={p[idx]:.4f}"+(f" ± {next(sdf):.4f}" if free[idx] else " (fixed)"))
    s=183105.0; lnlrs=p[n+1]+p[n+2]*np.log(s)+p[n+3]*np.log(2); lrs=np.exp(lnlrs); q1=p[n+5] if mode=='power' else 0
    Ps=np.exp(p[n+4]+q1*np.log(s)); aw=np.exp(p[n+6])*384**p[n+7]
    print(f"  P*(384B)={Ps:.3f}  pred 384B/512: lr*={lrs:.6f} wd*={Ps/(lrs*s):.5f}  a_w(384B)={aw:.4f}")
    for (b,t),i in gi.items():
        st=t*1e9/(b*4096); print(f"   group B={b} T={t}: L0={p[i]:.4f} lr*={np.exp(p[n+1]+p[n+2]*np.log(st)+p[n+3]*np.log(b/256)):.5f} wd*={np.exp(p[n+4]+q1*np.log(st))/(np.exp(p[n+1]+p[n+2]*np.log(st)+p[n+3]*np.log(b/256))*st):.4f} a_w={np.exp(p[n+6])*t**p[n+7]:.4f}")
    print("  residuals by run:"); 
    for rr,ri in zip(R,r): print(f"   B={rr['batch']} T={rr['tokens_B']} lr={rr['lr']} wd={rr['wd']} r={ri:+.4f}")
