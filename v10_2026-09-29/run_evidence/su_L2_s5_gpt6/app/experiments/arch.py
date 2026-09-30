import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,sys,json,time
s=np.load('/app/data/sample.npz');d=np.load('/app/data/dev.npz');Xs,ys=s['X'],s['y'];Xd,yd=d['X'],d['y']; base=json.load(open('/app/repo/config.json'))

def train(X,y,D,Y,hidden=256,act='relu',depth=2,skip=False,seed=0,wd=3,lr=.003,steps=1000):
 rng=np.random.default_rng(seed);mu=X.mean(0);sd=X.std(0)+1e-6;X=((X-mu)/sd).astype('float32');D=((D-mu)/sd).astype('float32'); dims=[32]+[hidden]*depth+[10];P={};m={};v={}
 for i in range(len(dims)-1): P[f'W{i}']=(rng.standard_normal((dims[i],dims[i+1]))*np.sqrt(2/dims[i])).astype('float32');P[f'b{i}']=np.zeros(dims[i+1],'float32');m[f'W{i}']=np.zeros_like(P[f'W{i}']);m[f'b{i}']=np.zeros_like(P[f'b{i}']);v[f'W{i}']=np.zeros_like(P[f'W{i}']);v[f'b{i}']=np.zeros_like(P[f'b{i}'])
 def fw(x,save=False):
  h=x; cs=[]
  for i in range(depth):
   a=h@P[f'W{i}']+P[f'b{i}']; h=np.tanh(a) if act=='tanh' else np.maximum(a,0) if act=='relu' else a/(1+np.exp(-a));cs.append((h,a))
  z=h@P[f'W{depth}']+P[f'b{depth}']; return z,cs
 def pred(x):return fw(x)[0]
 bs=128
 for t in range(steps):
  ix=rng.integers(len(X),size=bs);z,cs=fw(X[ix]);zz=z-z.max(1,keepdims=True);p=np.exp(zz);p/=p.sum(1,keepdims=True);dz=p;dz[np.arange(bs),y[ix]]-=1;dz/=bs; G={};dh=dz
  hprev=cs[-1][0];G[f'W{depth}']=hprev.T@dz;G[f'b{depth}']=dz.sum(0);dh=dz@P[f'W{depth}'].T
  for i in range(depth-1,-1,-1):
   h,a=cs[i]; hp=X[ix] if i==0 else cs[i-1][0]; da=dh*(1-(h*h)) if act=='tanh' else dh*(h>0) if act=='relu' else dh*(1/(1+np.exp(-a)) + a*np.exp(-a)/(1+np.exp(-a))**2)
   G[f'W{i}']=hp.T@da;G[f'b{i}']=da.sum(0);dh=da@P[f'W{i}'].T
  for n in P:
   m[n]=.9*m[n]+.1*G[n];v[n]=.999*v[n]+.001*G[n]**2;u=m[n]/(1-.9**(t+1))/(np.sqrt(v[n]/(1-.999**(t+1)))+1e-8)
   if n[0]=='W':u+=wd*P[n]
   P[n]-=(lr*u).astype('float32')
 return (pred(D).argmax(1)==Y).mean()
for act in ['relu','tanh','swish']:
 for dep in [1,2,3,4]:
  for h in [64,128,256]:
   a=train(Xs,ys,Xd,yd,h,act,dep,wd=3); print(act,dep,h,round(a,3),flush=True)
print('gold split')
for act in ['relu','tanh','swish']:
 for dep in [1,2,3]:
  a=train(Xd[:800],yd[:800],Xd[800:],yd[800:],128,act,dep,wd=3,steps=800); print(act,dep,round(a,3),flush=True)
