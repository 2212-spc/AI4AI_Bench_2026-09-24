from experiment import *
for N,D in [(5e7,2e9),(5e7,4e9),(5e7,8e9),(1e8,1e9),(2e8,1e9),(4e8,1e9)]:
 for w in [[.25]*4,[.1,.7,.1,.1],[.1,.1,.7,.1]]:
  run(w,N=N,D=D,tag='scaling')
