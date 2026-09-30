import numpy as np, json, csv, collections, itertools
def load():
    g=collections.defaultdict(list)
    for r in csv.DictReader(open('/app/notebook/runs.csv')):
        g[(float(r['N']),float(r['D']))].append((float(r['lr']),float(r['loss'])))
    for l in open('/app/lab_runs.jsonl'):
        r=json.loads(l); c=r['config']
        if r['status']=='ok': g[(c['N'],c['D'])].append((c['lr'],r['loss']))
    keys=sorted(g); N=[];D=[];LR=[];L=[];G=[]
    for i,k in enumerate(keys):
        for lr,loss in g[k]: N.append(k[0]);D.append(k[1]);LR.append(lr);L.append(loss);G.append(i)
    return keys,np.array(N),np.array(D),np.array(LR),np.array(L),np.array(G)
keys,N,D,LR,L,G=load()
lnN=np.log(N/3e7); lnD=np.log(D/3e9); lnlr=np.log(LR)
ng=len(keys)
def nm(f,x0,steps,iters=4000):
    n=len(x0); S=[np.array(x0,float)]
    for i in range(n):
        x=np.array(x0,float); x[i]+=steps[i]; S.append(x)
    S=np.array(S); F=np.array([f(x) for x in S])
    for _ in range(iters):
        o=np.argsort(F); S=S[o]; F=F[o]
        c=S[:-1].mean(0); xr=c+(c-S[-1]); fr=f(xr)
        if fr<F[0]:
            xe=c+2*(c-S[-1]); fe=f(xe)
            if fe<fr: S[-1],F[-1]=xe,fe
            else: S[-1],F[-1]=xr,fr
        elif fr<F[-2]: S[-1],F[-1]=xr,fr
        else:
            xc=c+0.5*(S[-1]-c); fc=f(xc)
            if fc<F[-1]: S[-1],F[-1]=xc,fc
            else:
                S[1:]=S[0]+0.5*(S[1:]-S[0]); F[1:]=[f(x) for x in S[1:]]
        if np.std(F)<1e-12: break
    o=np.argsort(F); return S[o[0]],F[o[0]]
def model_resid(theta, shape):
    # theta: c,a,b, then shape params (possibly N/D dependent curvature)
    c,a,b=theta[:3]; u=lnlr-(c+a*lnN+b*lnD)
    if shape=='quad': k=np.exp(theta[3]); pen=k*u**2
    elif shape=='asym': kl,kr=np.exp(theta[3]),np.exp(theta[4]); pen=np.where(u<0,kl,kr)*u**2
    elif shape=='asymN': kl,kr=np.exp(theta[3]),np.exp(theta[4]); p,q=theta[5],theta[6]; sc=np.exp(p*lnN+q*lnD); pen=sc*np.where(u<0,kl,kr)*u**2
    elif shape=='cubic': k,k3=np.exp(theta[3]),theta[4]; pen=k*u**2+k3*u**3
    elif shape=='expq': k,s=np.exp(theta[3]),theta[4]; pen=k*(np.exp(s*u)-1-s*u)/s**2  # asymmetric smooth
    elif shape=='expqN': k,s=np.exp(theta[3]),theta[4]; p,q=theta[5],theta[6]; sc=np.exp(p*lnN+q*lnD); pen=sc*k*(np.exp(s*u)-1-s*u)/s**2
    # per-group Lmin: linear solve
    y=L-pen; Lmin=np.array([y[G==i].mean() for i in range(ng)])
    return L-(Lmin[G]+pen), Lmin
shapes={'quad':([np.log(0.003),-0.3,-0.15,np.log(0.09)],[0.1,0.05,0.05,0.3]),
 'asym':([np.log(0.003),-0.3,-0.15,np.log(0.09),np.log(0.09)],[0.1,0.05,0.05,0.3,0.3]),
 'cubic':([np.log(0.003),-0.3,-0.15,np.log(0.09),0.01],[0.1,0.05,0.05,0.3,0.02]),
 'expq':([np.log(0.003),-0.3,-0.15,np.log(0.09),0.5],[0.1,0.05,0.05,0.3,0.2]),
 'asymN':([np.log(0.003),-0.3,-0.15,np.log(0.09),np.log(0.09),0,0],[0.1,0.05,0.05,0.3,0.3,0.1,0.1]),
 'expqN':([np.log(0.003),-0.3,-0.15,np.log(0.09),0.5,0,0],[0.1,0.05,0.05,0.3,0.2,0.1,0.1])}
out={}
for s,(x0,st) in shapes.items():
    f=lambda th: np.sum(model_resid(th,s)[0]**2)
    th,fv=nm(f,x0,st); th,fv=nm(f,th,np.array(st)*0.3)
    r,Lmin=model_resid(th,s); n=len(L); k=len(th)+ng
    print(s,'rms=%.4f'%np.sqrt(fv/(n-k)),'AIC=%.1f'%(n*np.log(fv/n)+2*k),'params',np.round(th,3))
    out[s]=(th,Lmin,r)
np.save('/app/fit_out.npy',{'keys':keys,'out':out},allow_pickle=True)
th,Lmin,r=out['expq']
for i,k in enumerate(keys): print('N=%.1e D=%.1e Lmin=%.4f  resid:'%(k[0],k[1],Lmin[i]),np.round(r[G==i],3))
