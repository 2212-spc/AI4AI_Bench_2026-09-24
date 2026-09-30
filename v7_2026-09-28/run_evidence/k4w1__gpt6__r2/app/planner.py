import heapq

def run(env):
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
  priority=(1-confidence)*pc
  if len(obs)<20:heapq.heappush(heap,(-priority,i))
 env.submit({q['id']:answer or '' for q,obs,answer,conf,comp in states})
