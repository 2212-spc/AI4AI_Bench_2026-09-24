import numpy as np
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X,y=s['X'],s['y'];D,Y=d['X'],d['y'];K=10
# methods evaluate using gold means from dev and crowd means sample, gaussian assumptions
for name,A,la,B,lb in [('crowd',X,y,D,Y),('gold',D,Y,X,y)]:
 mu=A.mean(0);sd=A.std(0); z=(A-mu)/sd; q=(B-mu)/sd
 means=np.array([z[la==k].mean(0) for k in range(K)])
 C=(z-means[la]).T@(z-means[la])/len(z)
 for reg in [0,.001,.003,.01,.03,.1]:
  Ci=np.linalg.inv(C+reg*np.eye(32)); s=q@Ci@means.T-.5*np.sum(means*(means@Ci),1)
  print(name,'lda',reg,(s.argmax(1)==lb).mean())
# estimate label confusion from nearest class center and compare
mu=D.mean(0);sd=D.std(0);z=(D-mu)/sd; q=(X-mu)/sd;means=np.array([z[Y==k].mean(0) for k in range(K)]);C=np.cov(z.T);ci=np.linalg.inv(C+.001*np.eye(32)); score=q@ci@means.T-.5*np.sum(means*(means@ci),1); pred=score.argmax(1)
print('crowd predicted using gold',np.bincount(pred,minlength=10)/len(pred)); print(np.array([[((pred==i)&(y==j)).sum() for j in range(K)] for i in range(K)]))
# sample labels with gold model, confusion probabilities
# holdout dev model validation via split
for frac in [.5,.8]:
 rng=np.random.default_rng(2); tr=rng.random(len(D))<frac; mu=D[tr].mean(0);sd=D[tr].std(0);zz=(D[tr]-mu)/sd;qq=(D[~tr]-mu)/sd; means=np.array([zz[Y[tr]==k].mean(0) for k in range(K)]); C=(zz-means[Y[tr]]).T@(zz-means[Y[tr]])/len(zz);ci=np.linalg.inv(C+.001*np.eye(32)); sc=qq@ci@means.T-.5*np.sum(means*(means@ci),1);print('dev split',frac,(sc.argmax(1)==Y[~tr]).mean())
