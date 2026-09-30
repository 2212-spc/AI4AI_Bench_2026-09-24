import json, random, collections
import numpy as np
import batchsim, policy
d={}
for f in ['data/dev_combinatorics.json','data/dev_number_theory.json','data/dev3_algebra.json','data/dev_geometry.json']:
    for q in json.load(open(f))['questions']: d.setdefault(q['topic'],[]).append(q)
batchsim.data=d
rng=random.Random(3); fails=collections.Counter(); used=collections.defaultdict(list); cases=[]
for r in range(8):
    env=batchsim.Env(rng, per_topic=100)
    orig=env.sample; log=collections.defaultdict(list)
    def sample(qid, v=0):
        a,s=orig(qid,v); log[qid].append((v,a,s)); return a,s
    env.sample=sample; policy.run(env)
    for q in env.questions:
        qid=q['id']; t=q['topic']; used[t].append(len(log[qid]))
        if env.result[qid]!=env.truth[qid]:
            seen=any(a==env.truth[qid] for v,a,s in log[qid])
            fails[(t, 'seen' if seen else 'unseen')]+=1
            if t=='algebra' and seen and len(cases)<6: cases.append((env.truth[qid], log[qid]))
print("fails per 800 q/topic:", dict(fails))
print("samples/q:", {t:round(np.mean(v),2) for t,v in used.items()}, "max:", {t:max(v) for t,v in used.items()})
for truth, lg in cases:
    print("truth", truth)
    for v,a,s in sorted(lg): print("   v%d %s %+.2f %s"%(v,a,s,'*' if a==truth else ''))
