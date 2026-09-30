import json, random, collections, policy
qs=sum([json.load(open('/app/long%d.json'%i))['questions'] for i in [1,2]],[])
class Env:
 def __init__(self,seed):
  self.questions=[{'id':q['id'],'topic':q['topic']} for q in qs]
  self.data={q['id']:{v:[s for s in q['samples'] if s['variant']==v] for v in range(5)} for q in qs}
  self.budget=6*len(qs); self.remaining=self.budget; self.n_variants=5; self.used=collections.Counter()
  rng=random.Random(seed)
  for vs in self.data.values():
   for samples in vs.values(): rng.shuffle(samples)
 def left(self): return self.remaining
 def sample(self,qid,v):
  self.remaining-=1; self.used[qid]+=1
  samples=self.data[qid][v]
  s=samples.pop(0); samples.append(s)
  return s['answer'],s['score']
 def submit(self,out): self.out=out
for seed in range(10):
 e=Env(seed); policy.run(e)
 correct=[q for q in qs if e.out[q['id']]==q['correct_answer']]
 print(seed,len(correct)/len(qs), 'topic', {t:round(sum(q['topic']==t for q in correct)/sum(q['topic']==t for q in qs),3) for t in sorted(set(q['topic'] for q in qs))},'max',max(e.used.values()))
 if seed==0:
  for q in qs:
   if e.out[q['id']]!=q['correct_answer']:
    sampled=[s for v,ss in e.data[q['id']].items() for s in ss[-sum(1 for _ in []):]] if False else []
    print(q['id'],q['topic'],'used',e.used[q['id']],'available correct',sum(s['correct'] for s in q['samples']))
