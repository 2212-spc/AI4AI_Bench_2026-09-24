from experiment import *
for N in [1e8,2e8,4e8]:run([.25]*4,N=N)
for D in [2e9,4e9,8e9,16e9]:run([.25]*4,D=D)
for i in range(4):
 for epoch in [2,4,10,30]:
  p=np.ones(4)*1e9;p[i]=1e9/epoch
  run(np.eye(4)[i],pool=p)
for i in range(4):run(np.eye(4)[i],D=4e9)
