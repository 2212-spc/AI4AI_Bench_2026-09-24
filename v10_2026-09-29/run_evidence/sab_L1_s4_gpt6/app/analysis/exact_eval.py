from optimize import *
import json
sched=[float(x.split()[1]) for x in (ROOT/'rollout.conf').read_text().splitlines()]
def exact_value(fs):
    vals=[];tw=[]
    for h,f in enumerate(fs):
      for l,wgt in zip(rates[h],weights):
        c=int(np.clip(np.ceil(l*(sa+(sb-sa)*f)/.75),4,16))
        a,w=hyperexp(float(l),float(f),c,th,sa,sb)
        vals.append((1-a)*(ra+(rb-ra)*f)-.008*w);tw.append(l*wgt)
    return np.average(vals,weights=tw)
base=exact_value([0]*24);q1=exact_value([1]*24)-base;q2=exact_value(sched)-base
print('exact base q1 q2',base,q1,q2)
# schedule neighbor exact around peak and changing f grid
for h in range(10,21):
 f=sched[h];cand=np.round(np.arange(max(0,f-.03),min(1,f+.031),.01),2);v=[]
 for x in cand:
  ss=sched.copy();ss[h]=x;v.append(exact_value(ss))
 print(h,f,cand[np.argmax(v)],max(v)-exact_value(sched))
print(json.dumps({'base':base,'Q1':q1,'Q2':q2}))
