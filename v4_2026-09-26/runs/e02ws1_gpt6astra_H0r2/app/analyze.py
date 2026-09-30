import json
import numpy as np
rows=[json.loads(l) for l in open('/app/notebook/runs.jsonl')]+json.load(open('/app/measured_runs.json'))['runs']
# Fit the geometric acceptance curve to all measured positions, weighting
# duration and binomial variance. Grid refinement avoids scipy dependency.
a0,r0=.91,.945
for span,step in [(.02,.0001),(.0002,.000001)]:
 aa=np.arange(max(.8943,a0-span),min(.9547,a0+span)+step/2,step)
 rr=np.arange(max(.9366,r0-span),min(.9735,r0+span)+step/2,step)
 a,r=np.meshgrid(aa,rr,indexing='ij');loss=np.zeros_like(a)
 for row in rows:
  c=row['config']
  for i,obs in enumerate(row.get('accept_by_position',[])):
   p=a*r**i
   loss-=c['dur']*(obs*np.log(p)+(1-obs)*np.log1p(-p))
 ind=np.unravel_index(np.argmin(loss),loss.shape)
 a0,r0=float(a[ind]),float(r[ind])
print('acceptance MLE',a0,r0,'p',a0*r0**np.arange(4))
from collections import defaultdict
groups=defaultdict(list)
for row in rows:
 c=row['config'];groups[c['seq'],c['batch'],c['spec_g']].append(row)
var=np.zeros(2);df=0
for key,rr in groups.items():
 if len(rr)<2:continue
 dur=np.array([x['config']['dur']/20 for x in rr]);xs=np.array([[np.log(x['tokens_per_s']),np.log(1000*x['batch']/x['ms_per_token'])] for x in rr])
 avg=np.average(xs,axis=0,weights=dur)
 var+=np.sum(dur[:,None]*(xs-avg)**2,axis=0);df+=len(rr)-1
print('empirical reference relative sd',np.sqrt(var/df),'throughput weight',var[1]/sum(var))
# Use a stable 2:1 precision ratio for throughput vs reciprocal ms/token.
w_tps=2/3
summary={}
for key,rr in groups.items():
 dur=np.array([x['config']['dur'] for x in rr]);xs=np.array([[np.log(x['tokens_per_s']),np.log(1000*x['batch']/x['ms_per_token'])] for x in rr])
 means=np.average(xs,axis=0,weights=dur)
 tps=np.exp(w_tps*means[0]+(1-w_tps)*means[1]); summary[key]=tps
 print(key,'n',len(rr),'tps and inverse ms',np.exp(means),'estimate',tps)
e4=1+np.cumprod(a0*r0**np.arange(4)).sum()
speed=summary[1536,113,4]/summary[1536,113,0]
c=(e4/speed-1)/4
frac=4*c/(1+4*c)
print('E4',e4,'speed',speed,'c',c,'fraction',frac)
for tail in [.85,1]:
 p4=a0*r0**np.arange(4); p8=np.r_[p4,p4[-1]*tail**np.arange(1,5)]
 e8=1+np.cumprod(p8).sum();gain=e8/e4*(1+4*c)/(1+8*c)-1
 print('tail',tail,'gain',gain)
print('q2 endpoints',summary[1446,53,0],summary[883,88,0])
# Check individual acceptance residuals under several plausible sample sizes.
for n20 in [500,1000,2000]:
 maxz=0;ss=0;n=0
 for row in rows:
  for i,obs in enumerate(row.get('accept_by_position',[])):
   p=a0*r0**i;sd=np.sqrt(p*(1-p)/(n20*row['config']['dur']/20));z=(obs-p)/sd
   maxz=max(maxz,abs(z));ss+=z*z;n+=1
 print('n20',n20,'max z',maxz,'sum squares',ss,'dof',n)
answers={'q1':{'lo':round(frac,6),'hi':round(frac,6)},'q2':{'lo':round(summary[1446,53,0],3),'hi':round(summary[883,88,0],3)},'q3':{'verdict':'refutable','witness':{'a':round(a0,6),'rho':round(r0,6),'rho_tail':.85}},'q4':{'verdict':'entailed'}}
with open('/app/answers.json','w') as f:json.dump(answers,f,indent=2);f.write('\n')
