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
 gs=groups(obs,topic);ans=list(gs)+[None]
 mat=[likelihoods(gs,a) for a in ans]
 vals=[sum(x*w for x,w in zip(row,weights)) for row in mat]
 tot=sum(vals)
 if not tot:return list(gs)[0],.01, [1/len(weights)]*len(weights)
 i=max(range(len(ans)-1),key=lambda i:vals[i]);prob=vals[i]/tot
 comp=[sum(row[j] for row in mat)*weights[j]/tot for j in range(len(weights))]
 return ans[i],prob,comp
