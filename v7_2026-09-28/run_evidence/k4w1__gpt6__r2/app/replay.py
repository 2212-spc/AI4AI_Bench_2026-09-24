import json,random,collections,sys,time
import policy
class Env:
 def __init__(self,qs,rng):
  self.questions=[{'id':str(i),'topic':q['topic']} for i,q in enumerate(qs)];self.budget=6*len(qs);self.n_variants=5;self.used=0;self.qs=qs;self.samples={};self.counts=collections.Counter()
  for i,q in enumerate(qs):
   for v in range(5):
    ss=[s for s in q['samples'] if s['variant']==v];rng.shuffle(ss);self.samples[str(i),v]=ss
 def left(self):return self.budget-self.used
 def sample(self,qid,v):
  self.used+=1;self.counts[qid]+=1;s=self.samples[qid,v].pop();return s['answer'],s['score']
 def submit(self,out):self.out=out
qs={t:sum([json.load(open('/app/'+t+f+'.json'))['questions'] for f in (['','2'] if t in ('algebra','geometry') else [''])],[]) for t in ['algebra','geometry','combinatorics','number_theory']}
acc=[]
for seed in range(int(sys.argv[1]) if len(sys.argv)>1 else 5):
 rng=random.Random(seed);qq=sum([rng.sample(q,40) for q in qs.values()],[]);env=Env(qq,rng);t=time.time();policy.run(env)
 good=[env.out[str(i)]==q['correct_answer'] for i,q in enumerate(qq)];acc.append(sum(good)/len(good))
 print(seed,round(acc[-1],4),{topic:(round(sum(good[i] for i,q in enumerate(qq) if q['topic']==topic)/40,3),round(sum(env.counts[str(i)] for i,q in enumerate(qq) if q['topic']==topic)/40,2)) for topic in qs},round(time.time()-t,2),flush=True)
 if seed==0:
  for i,q in enumerate(qq):
   if good[i]:continue
   print('FAIL',q['topic'],env.counts[str(i)],'answer',env.out[str(i)],'true',q['correct_answer'])
print('MEAN',sum(acc)/len(acc))
