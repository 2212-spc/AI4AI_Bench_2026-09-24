import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,json,time
a=np.load('/app/data/sample.npz'); v=np.load('/app/data/dev.npz'); X=a['X']; y=a['y']; mu=X.mean(0); sd=X.std(0); X=(X-mu)/sd; V=(v['X']-mu)/sd; Y=np.eye(10)[y].astype(np.float32);
ii,jj=np.triu_indices(X.shape[1]);
def feat(x):
 q=x[:,ii]*x[:,jj]; q[:,ii==jj]=(q[:,ii==jj]-1)/np.sqrt(2); return np.column_stack([np.ones(len(x)),x,q]).astype(np.float64)
F=feat(X); D=feat(V); G=F.T@F; H=F.T@Y; rows=[]
for linear in [0.01,0.1,1,10]:
 for quad in [0.01,0.1,0.5,1,3,10,30]:
  reg=np.array([0.0001]+[linear]*32+[quad]*len(ii))*len(y); W=np.linalg.solve(G+np.diag(reg),H); pred=(D@W).argmax(1); acc=np.mean(pred==v['y']); rows.append((acc,linear,quad));
print('quadratic ridge',sorted(rows,reverse=True)[:15]);
for reg in [0.001,.01,.1,1]:
 W=np.linalg.solve(G[:33,:33]+np.diag([.01]+[reg*len(y)]*32),H[:33]); print('linear',reg,np.mean((D[:,:33]@W).argmax(1)==v['y']))
