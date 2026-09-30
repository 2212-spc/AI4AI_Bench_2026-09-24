import numpy as np
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];K=10
# standardize
mu=X.mean(0);sd=X.std(0);Z=(X-mu)/sd; Q=(D-mu)/sd
for feats in [np.arange(32),np.array([1,2,3,4,7,8,9,12,15,18,19,22,25,27,29,30])]:
 z=Z[:,feats];q=Q[:,feats]
 print('features',len(feats))
 for reg in [.01,.1,.3,1,3,10,30,100]:
  # gaussian diag
  means=np.array([z[y==k].mean(0) for k in range(K)]); vars=np.array([z[y==k].var(0)+reg for k in range(K)])
  logp=-.5*(((q[:,None,:]-means)**2)/vars+np.log(vars)).sum(2)+np.log(np.bincount(y,minlength=K)/len(y))
  print('diag',reg,round((logp.argmax(1)==Y).mean(),3),end='; ')
 print()
 # shared covariance LDA
 means=np.array([z[y==k].mean(0) for k in range(K)]); C=(z-means[y]).T@(z-means[y])/len(z)
 for reg in [.01,.1,.3,1,3,10]:
  Ci=np.linalg.inv(C+reg*np.eye(len(feats))); score=q@Ci@means.T-.5*np.sum(means*(means@Ci),1)+np.log(np.bincount(y,minlength=K)/len(y)); print('lda',reg,round((score.argmax(1)==Y).mean(),3),end='; ')
 print()
 # class cov qda
 for reg in [.1,.3,1,3,10]:
  scores=[]
  for k in range(K):
   C=np.cov(z[y==k].T)+reg*np.eye(len(feats)); sign,ld=np.linalg.slogdet(C); ci=np.linalg.inv(C); dm=q-means[k]; scores.append(-.5*(np.sum(dm@ci*dm,1)+ld)+np.log((y==k).mean()))
  print('qda',reg,round((np.array(scores).argmax(0)==Y).mean(),3),end='; ')
 print()
# nearest neighbors brute
for norm in ['raw','std']:
 a=X if norm=='raw' else Z; b=D if norm=='raw' else Q
 for k in [1,3,5,10,20,50,100]:
  out=[]
  for i in range(0,len(b),100):
   ix=np.argpartition(((b[i:i+100,None,:]-a[None,:,:])**2).sum(2),k,axis=1)[:,:k];
   out.append(np.array([np.bincount(y[j],minlength=K).argmax() for j in ix]))
  p=np.concatenate(out);print(norm,k,(p==Y).mean())
