import numpy as np
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];K=10
mu=X.mean(0);sd=X.std(0);X=(X-mu)/sd;D=(D-mu)/sd
# shared covariance EM, diagonal or full
for typ in ['diag','full']:
 for init in ['crowd','random']:
  for reg in [.01,.03,.1,.3]:
   if init=='crowd': M=np.array([X[y==k].mean(0) for k in range(K)]); pi=np.bincount(y,minlength=K)/len(y)
   else: M=X[np.random.default_rng(0).choice(len(X),K,replace=False)];pi=np.ones(K)/K
   for it in range(50):
    if typ=='full':
     C=(X-M[np.argmax(-((X[:,None,:]-M[None,:,:])**2).sum(2),1)]).T@ (X-M[np.argmax(-((X[:,None,:]-M[None,:,:])**2).sum(2),1)])/len(X) # rough
     # use identity initially? weighted expensive
    if typ=='diag':
     V=np.zeros((K,32))
     # responsibility based on diagonal
     ll=-.5*(((X[:,None,:]-M[None,:,:])**2)/(V+reg) if it else ((X[:,None,:]-M[None,:,:])**2)/1).sum(2)
    else: break
    if it==0: pass
    # recompute with current pooled diagonal
    if it>0: pass
    # proper use pooled covariance based responsibilities previous
    if it==0: R=np.exp(ll-ll.max(1)[:,None]);R/=R.sum(1)[:,None]
    else:
     ll=-.5*(((X[:,None,:]-M[None,:,:])**2)/(V+reg)).sum(2)-.5*np.log(V+reg).sum(1)[None,:]+np.log(pi)[None,:]
     R=np.exp(ll-ll.max(1)[:,None]);R/=R.sum(1)[:,None]
    Nk=R.sum(0);M=R.T@X/Nk[:,None];pi=Nk/len(X);V=np.einsum("nk,nkd->kd",R,(X[:,None,:]-M[None,:,:])**2)/Nk[:,None]
   # classify dev comp
   ll=-.5*(((D[:,None,:]-M[None,:,:])**2)/(V+reg)).sum(2)-.5*np.log(V+reg).sum(1)[None,:]+np.log(pi)[None,:]; comp=ll.argmax(1)
   # map comp->label based on crowd labels responsibilities
   # score each comp and crowd y using R on train
   cm=np.zeros((K,K)); cm=R.T@np.eye(K)[y]
   mapc=cm.argmax(1); pred=mapc[comp]; print(typ,init,reg,'crowdmap', (pred==Y).mean(), 'comp',mapc, 'pi',np.round(pi,2),flush=True)
