import estimator,random,json,statistics,time
rows=json.load(open('dev.json'))['prompts']
class Env:
 label_budget=220
 target_coverage=.90
 def __init__(self,seed):
  r=random.Random(seed);self.rows=rows;self.topics=['code','math','writing','factual'];self.lengths=['short','long'];ss=sorted(set(x['topic']+'|'+x['length'] for x in rows));ws=[r.gammavariate(3,1) for s in ss];self.production_mix={s:w/sum(ws) for s,w in zip(ss,ws)};self.truth={};self.used=set()
  for x in rows:
   k=(x['topic'],x['length'],x['judge_win']);rate=.012
   if k==('writing','short',False):rate=.07
   if k==('math','long',True):rate=.15
   if k==('math','short',False):rate=.07
   if k==('code','long',True):rate=.07
   if k==('code','short',True):rate=.045
   if k==('factual','short',False):rate=.065
   self.truth[str(x['id'])]=bool(x['judge_win'])!=(r.random()<rate)
  self.true=sum(self.production_mix[s]*sum(self.truth[str(x['id'])] for x in rows if x['topic']+'|'+x['length']==s)/sum(x['topic']+'|'+x['length']==s for x in rows) for s in ss)
 def label(self,ids):
  self.used.update(ids);assert len(self.used)<=220
  return {str(i):self.truth[str(i)] for i in ids}
 def submit(self,p,lo,hi):self.result=(p,lo,hi)
res=[]
for seed in range(12):
 e=Env(seed);estimator.estimate(e);p,l,h=e.result;res.append((h-l,l<=e.true<=h,p-e.true));print(seed,tuple(round(v,5) for v in [e.true,p,l,h,h-l]),l<=e.true<=h,flush=True)
print('meanwidth coverage bias',*[statistics.mean(x[j] for x in res) for j in range(3)])
