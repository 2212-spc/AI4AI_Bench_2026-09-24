from optimize import *
import numpy as np
sched=[float(x.split()[1]) for x in (ROOT/'rollout.conf').read_text().splitlines()]
def ev(fs):
 num=den=0
 for h,f in enumerate(fs):
  for l,ww in zip(rates[h],weights):
   c=int(np.clip(np.ceil(l*(sa+(sb-sa)*f)/.75),4,16));a,w=hyperexp(float(l),float(f),c,th,sa,sb);num+=l*ww*((1-a)*(ra+(rb-ra)*f)-.008*w);den+=l*ww
 return num/den
base=ev([0]*24); print('base',base,flush=True); q1=ev([1]*24)-base;print('q1',q1,flush=True);q2=ev(sched)-base;print('q2',q2,flush=True)
