from experiment import *
w=np.load('/app/optimum.npy')
run(w,N=4e8,D=4e9,pool=U*4e9/5e11,tag='targetvalidate_large')
run(w,N=5e7,D=8e9,pool=U*8e9/5e11,tag='targetvalidate_long')
run(w,pool=U/500,tag='targetvalidate')
for a,b in [(0,1),(0,2),(0,3)]:
 for sign in [-1,1]:
  v=w.copy();v[a]+=.08*sign;v[b]-=.08*sign
  run(v,pool=U/500,tag='targetneighbor')
