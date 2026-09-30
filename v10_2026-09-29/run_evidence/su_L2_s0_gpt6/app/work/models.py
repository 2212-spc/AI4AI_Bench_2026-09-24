import numpy as np
A=np.load('/app/data/sample.npz');D=np.load('/app/data/dev.npz'); X=A['X'];y=A['y']; V=D['X'];z=D['y'];
# standard
mu=X.mean(0);sd=X.std(0); X=(X-mu)/sd;V=(V-mu)/sd
for p in [1,2,3]:
 F=X**p; Q=V**p
 # class means weighted; nearest centroid
 means=np.array([F[y==c].mean(0) for c in range(10)])
 print('centroid',p,np.mean(((Q[:,None,:]-means[None,:,:])**2).sum(2).argmin(1)==z))
# kNN
for k in [1,3,5,10,20,40,80]:
 pred=[]
 for i in range(0,len(V),100):
  dist=((V[i:i+100,None,:]-X[None,:,:])**2).sum(2); ii=np.argpartition(dist,k,axis=1)[:,:k]; votes=np.zeros((len(ii),10)); np.add.at(votes,(np.arange(len(ii))[:,None],y[ii]),1);pred.extend(votes.argmax(1))
 print('knn',k,np.mean(np.array(pred)==z))
# gaussian diagonal / qda
for smooth in [.01,.1,.5,1,2,5]:
 means=np.array([X[y==c].mean(0) for c in range(10)]); var=np.array([X[y==c].var(0)+smooth for c in range(10)]); prior=np.bincount(y,minlength=10)/len(y)
 ll=-.5*((V[:,None,:]-means)**2/var+np.log(var)).sum(2)+np.log(prior)
 print('qda',smooth,np.mean(ll.argmax(1)==z))
