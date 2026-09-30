import math, random
import sched
class Env:
 n_segments=40;b_min=4;b_max=8192;t_overhead=1.0194;throughput=576
 def __init__(self,c,j,s,seed):
  self.c=c;self.j=j;self.s=s;self.i=0;self.total=0;self.bs=[];self.opt=0;self.rng=random.Random(seed)
 def run_segment(self,b):
  assert self.b_min<=b<=self.b_max
  a=1.113+.0127*self.i+(self.j+self.s*(self.i-self.c) if self.i>=self.c else 0)
  noise=math.exp(self.rng.gauss(-.5*.07**2,.07))
  steps=(.02518+a/b)*noise
  sec=steps*(self.t_overhead+b/self.throughput)
  optb=math.sqrt(a*self.t_overhead*self.throughput/.02518)
  self.opt+=(.02518+a/optb)*(self.t_overhead+optb/self.throughput)*noise
  self.i+=1;self.total+=sec;self.bs.append(b)
  return {'steps':steps}
for c,j,s in [(25,4.7,.03),(16,.8,.01),(32,10,.1),(20,20,.3),(28,3,.4),(31,.4,.05)]:
 e=Env(c,j,s,100+c);sched.run(e);assert e.i==40
 print(c,j,s,'total',round(e.total,4),'oracle',round(e.opt,4),'gap',round(e.total-e.opt,4),'batches',e.bs)
