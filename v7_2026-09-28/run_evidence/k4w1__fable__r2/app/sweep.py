import json, glob, random, importlib, collections, sys, itertools
import numpy as np
import batchsim, policy
def load(files):
    d={}
    for f in files:
        for q in json.load(open(f))['questions']: d.setdefault(q['topic'],[]).append(q)
    return d
common=['data/dev_combinatorics.json','data/dev_number_theory.json']
sets={'A3+G1': load(common+['data/dev3_algebra.json','data/dev_geometry.json']),
      'A12+G2': load(common+['data/dev_algebra.json','data/dev2_algebra.json','data/dev2_geometry.json'])}
def run(params, reps=6):
    for k,v in params.items(): setattr(policy,k,v)
    res={}
    for name,d in sets.items():
        batchsim.data=d; rng=random.Random(7); tot=collections.defaultdict(list)
        for r in range(reps):
            env=batchsim.Env(rng, per_topic=100); policy.run(env)
            for q in env.questions: tot[q['topic']].append(env.result[q['id']]==env.truth[q['id']])
        res[name]={t:round(np.mean(v),3) for t,v in tot.items()}
        res[name]['all']=round(np.mean([x for v in tot.values() for x in v]),3)
    return res
if __name__=="__main__":
  grid=dict(VARIANT_BONUS=[1.0,1.5,2.5], REPEAT_WEIGHT=[0.2,0.35,0.6], DONE_CONF=[6.0,8.0,10.0])
  rows=[]
  for combo in itertools.product(*grid.values()):
      p=dict(zip(grid.keys(),combo)); p["EARLY_CONF"]=p['DONE_CONF']
      r=run(p); score=np.mean([r[s]['all'] for s in r])
      rows.append((score,p,r)); print(round(score,4), p, r, flush=True)
  rows.sort(key=lambda x:-x[0]); print("BEST", rows[0])
