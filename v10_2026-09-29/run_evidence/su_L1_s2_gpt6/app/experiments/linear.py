import numpy as np
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');X=s['X'];y=s['y'];T=d['X'];ty=d['y'];
mu=X.mean(0);sd=X.std(0);X=(X-mu)/sd;T=(T-mu)/sd
for reg in [0.001,.01,.1,1,10,100]:
 # ridge onehot
 A=X.T@X+reg*np.eye(X.shape[1]); W=np.linalg.solve(A,X.T@np.eye(10)[y]);
 print(reg,np.mean((T@W).argmax(1)==ty))
for temp in [0.1,.3,.5,1,2,3]:
 # class centroids nearest
 c=np.array([X[y==k].mean(0) for k in range(10)])
 print('cent',temp,np.mean((-(T[:,None,:]-c[None,:,:])**2).sum(2).argmax(1)==ty))
