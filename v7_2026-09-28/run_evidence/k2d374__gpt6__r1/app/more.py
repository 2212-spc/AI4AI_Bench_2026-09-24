from experiment import *
for i in range(4):
 for epoch in [2,5,20,100]:
  pool=[1e10]*4;pool[i]=1e9/epoch
  run(np.eye(4)[i],pool=pool)
rng=np.random.default_rng(45)
for j in range(16):
 run(rng.dirichlet([1]*4),D=[1e9,2e9,4e9,1e9][j%4],N=[5e7,5e7,5e7,2e8][j%4])
for N in [1e8,2e8,4e8]: run([.25]*4,D=4e9,N=N)
