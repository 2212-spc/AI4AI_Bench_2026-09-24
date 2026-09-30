from joint import *
p=np.load('/app/joint.npy');rng=np.random.default_rng(74)
fitted=pred(p,X,N,U);res=Y-fitted
vals=[]
m=np.array([.70161,.12302,.09244,.08293])
for j in range(150):
 y=fitted+res[rng.integers(0,len(res),len(res))]
 pp,c=lm(lambda pp:(pred(pp,X,N,U)-y).ravel(),p,100)
 vals.append((pred(pp,(500*m)[None,:],np.array([50]),U0[None,:])@np.array([.121,.256,.623])).item())
print('bootstrap target quantiles',np.quantile(vals,[.005,.025,.5,.975,.995]))
print('target epochs',m*500/U0)
