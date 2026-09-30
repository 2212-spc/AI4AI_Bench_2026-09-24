"""Adaptive pooled sampling using a calibrated latent-error model.

Candidates can be correct, independent errors, a question-wide recurrent error,
or a template-specific recurrent error.  Integrating these alternatives keeps
high-scoring traps from dominating the decision.  All calibration constants
were estimated from labelled development samples; runtime uses only stdlib.
"""

import math
# Each component: topology, five (correct, trap, fresh-error) rates, weight.
def components():
 out=[]
 for p in [.06,.15,.28,.42,.58,.73,.86]:out.append(('noise',[(p,0,1-p)]*5))
 for mode in ['shared','local']:
  for p in [.06,.14,.24,.36,.5]:
   out.append((mode,[(p,.8-p,.2)]*5))
 # A question may have a particularly effective paraphrase.
 for p in [.08,.2,.35]:
  for v in range(5):
   rates=[(p,.8-p,.2)]*5;rates[v]=(.68,.12,.2)
   out.append(('shared',rates))
 return out
COMPS=components()

def groups(obs,topic):
 trapmu={'algebra':1.65,'geometry':1.8}.get(topic,-.6)
 out={}
 for v,a,s in obs:
  dc=math.exp(-.5*((s-1)/.4)**2)
  dt=math.exp(-.5*((s-trapmu)/.4)**2)
  dn=math.exp(-.5*((s+.65)/.4)**2)
  if topic in ('algebra','geometry'):dn=.94*dn+.06*dt
  z=max(dc,dt,dn,1e-200);dc/=z;dt/=z;dn/=z
  if a not in out:out[a]=[[0]*5,1.,1.,1.]
  g=out[a];g[0][v]+=1;g[1]*=max(dc,1e-100);g[2]*=max(dt,1e-100);g[3]*=max(dn,1e-100)
 return out

def likelihoods(gs,c):
 cg=gs.get(c,[[0]*5,1.,1.,1.]);cc=cg[0];rc=cg[1]
 wrong=[g for a,g in gs.items() if a!=c]
 # Precompute valid assignments of the shared trap.
 repeats=[g for g in wrong if sum(g[0])>1]
 shared=[]
 if len(repeats)<2:
  choices=[None]+wrong if not repeats else repeats
  for t in choices:
   nc=[0]*5;nt=t[0] if t else [0]*5;density=rc*(t[2] if t else 1.)
   for g in wrong:
    if g is t:continue
    density*=g[3]
    for v in range(5):nc[v]+=g[0][v]
   shared.append((nt,nc,density))
 # Local traps cannot recur across templates.
 local=None
 if all(sum(x>0 for x in g[0])==1 for g in wrong):
  local=[]
  for v in range(5):
   ww=[g for g in wrong if g[0][v]];rep=[g for g in ww if g[0][v]>1];terms=[]
   if len(rep)<2:
    choices=[None]+ww if not rep else rep
    for t in choices:
     nt=t[0][v] if t else 0;nn=0;d=t[2] if t else 1.
     for g in ww:
      if g is t:continue
      nn+=g[0][v];d*=g[3]
     terms.append((nt,nn,d))
   local.append(terms)
 result=[]
 noise_density=rc
 for g in wrong:noise_density*=g[3]
 nn=sum(sum(g[0]) for g in wrong);ncc=sum(cc)
 for mode,rates in COMPS:
  if mode=='noise':
   p,_,pn=rates[0]
   val=0 if repeats else noise_density*p**ncc*pn**nn
  elif mode=='shared':
   val=0.
   for nt,nnoise,d in shared:
    x=d
    for v,(pc,pt,pn) in enumerate(rates):x*=pc**cc[v]*pt**nt[v]*pn**nnoise[v]
    val+=x
  else:
   val=rc
   if local is None:val=0.
   else:
    for v,(pc,pt,pn) in enumerate(rates):
     val*=pc**cc[v]*sum(d*pt**nt*pn**nn for nt,nn,d in local[v])
  result.append(val)
 return result

def posterior(obs,topic,weights):
 """Return best observed answer, posterior confidence, and type posterior."""
 gs=groups(obs,topic);ans=list(gs)+[None]
 mat=[likelihoods(gs,a) for a in ans]
 vals=[sum(x*w for x,w in zip(row,weights)) for row in mat]
 tot=sum(vals)
 if not tot:return list(gs)[0],.01, [1/len(weights)]*len(weights)
 i=max(range(len(ans)-1),key=lambda i:vals[i]);prob=vals[i]/tot
 comp=[sum(row[j] for row in mat)*weights[j]/tot for j in range(len(weights))]
 return ans[i],prob,comp

WEIGHTS = [0.007830173328568061, 0.029240781405593715, 0.03944710933185396, 0.16603886343794666, 0.07780659405427497, 0.08772949153997914, 0.006092790658241845, 0.020647457038002442, 0.0284456438161816, 0.10741912219563458, 0.011380160898993527, 0.003771164416248011, 0.0538713015836155, 0.011331768590970688, 0.21305046284986817, 0.023862430017571254, 0.002082870415506199, 0.0015675571711557088, 0.0015675571711557088, 0.0015675571711557088, 0.0015675571711557088, 0.0015675571711557088, 0.0038930582560120263, 0.0038930582560120263, 0.0038930582560120263, 0.0038930582560120263, 0.0038930582560120263, 0.016529747457022202, 0.016529747457022202, 0.016529747457022202, 0.016529747457022202, 0.016529747457022202]
import heapq

def run(env):
 # Initial priority guarantees every question is sampled before any repeats.
 states=[];heap=[]
 for i,q in enumerate(env.questions):
  states.append([q,[],None,0.,None])
  heapq.heappush(heap,(-1.,i))
 while env.left()>0 and heap:
  _,i=heapq.heappop(heap)
  q,obs,answer,confidence,comp=states[i]
  counts=[sum(v==j for v,a,s in obs) for j in range(env.n_variants)]
  v=min(range(env.n_variants),key=lambda j:(counts[j],(j-i)%env.n_variants))
  a,s=env.sample(q['id'],v);obs.append((v,a,s))
  answer,confidence,comp=posterior(obs,q['topic'],WEIGHTS)
  states[i]=[q,obs,answer,confidence,comp]
  pc=sum(w*sum(r[0] for r in rates)/5 for w,(mode,rates) in zip(comp,COMPS))
  # Approximate marginal value: residual error times discovery probability.
  priority=(1-confidence)*pc
  if len(obs)<40:heapq.heappush(heap,(-priority,i))
 env.submit({q['id']:answer or '' for q,obs,answer,conf,comp in states})
