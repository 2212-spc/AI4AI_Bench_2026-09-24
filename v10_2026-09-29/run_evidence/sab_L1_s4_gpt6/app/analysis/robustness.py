from optimize import *
sched=np.array([float(x.split()[1]) for x in (ROOT/'rollout.conf').read_text().splitlines()]);grid=np.linspace(0,1,251)
for scale in [.99,1,1.01]:
 total_best=total_chosen=total=0
 for h in range(24):
  l=rates[h,None,:]*scale;ss=scores(l,grid[:,None]);vs=(ss*l*weights).sum(axis=1)
  chosen=float(np.sum(scores(l,sched[h])*l*weights));total_best+=vs.max();total_chosen+=chosen;total+=np.sum(l*weights)
 print('scale',scale,'proxy optimality gap',(total_best-total_chosen)/total,flush=True)
