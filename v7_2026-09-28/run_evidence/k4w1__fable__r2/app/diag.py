import json, glob, random, importlib, collections, sys
import numpy as np
import batchsim, policy
files = sys.argv[1:] 
data={}
for f in files:
    for q in json.load(open(f))['questions']: data.setdefault(q['topic'],[]).append(q)
batchsim.data=data
rng=random.Random(1); tot=collections.defaultdict(list); used=collections.defaultdict(list); fails=[]
for r in range(10):
    env=batchsim.Env(rng, per_topic=100)
    # wrap sample to count per topic
    topic_of={q['id']:q['topic'] for q in env.questions}
    cnt=collections.Counter(); orig=env.sample
    def sample(qid, variant=0):
        cnt[qid]+=1; return orig(qid, variant)
    env.sample=sample
    policy.run(env)
    for q in env.questions:
        t=q['topic']; ok=env.result[q['id']]==env.truth[q['id']]
        tot[t].append(ok); used[t].append(cnt[q['id']])
print({t:round(np.mean(v),3) for t,v in tot.items()})
print("mean samples/q", {t:round(np.mean(v),2) for t,v in used.items()})
