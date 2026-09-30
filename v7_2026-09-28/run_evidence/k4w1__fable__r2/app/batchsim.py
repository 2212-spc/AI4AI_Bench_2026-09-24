import json, glob, random, importlib, sys
import numpy as np
data = {}
for f in glob.glob('data/dev*.json'):
    d = json.load(open(f))
    for q in d['questions']:
        data.setdefault(q['topic'], []).append(q)

class Env:
    def __init__(self, rng, per_topic=100, budget_per=6):
        self.questions=[]; self.pools={}; self.truth={}
        for t, qs in data.items():
            for i in range(per_topic):
                q=rng.choice(qs); qid=f"{t[:2]}-{i}"
                self.questions.append({"id":qid,"topic":t})
                by_v={v:[s for s in q['samples'] if s['variant']==v] for v in range(5)}
                for v in by_v: rng.shuffle(by_v[v])
                self.pools[qid]=by_v; self.truth[qid]=q['correct_answer']
        rng.shuffle(self.questions)
        self.budget=budget_per*len(self.questions); self.used=0; self.n_variants=5
        self.topics=list(data); self.result=None; self.exhaust=0
    def left(self): return self.budget-self.used
    def sample(self, qid, variant=0):
        if self.used>=self.budget: raise RuntimeError("pool empty")
        self.used+=1
        p=self.pools[qid][variant]
        if not p:
            # sim limitation: variant exhausted -> fall back to any other variant's leftover
            self.exhaust+=1
            for v in range(5):
                if self.pools[qid][v]: p=self.pools[qid][v]; break
            if not p: return "EXH", -2.0
        s=p.pop(); return s['answer'], s['score']
    def submit(self, out): self.result=out

def run(reps=5, seed=0, verbose=True):
    import policy; importlib.reload(policy)
    rng=random.Random(seed); accs={}; tot=[]; used=[]; ex=0
    for r in range(reps):
        env=Env(rng); policy.run(env)
        for q in env.questions:
            ok = env.result.get(q['id'])==env.truth[q['id']]
            accs.setdefault(q['topic'],[]).append(ok); tot.append(ok)
        used.append(env.used); ex+=env.exhaust
    if verbose:
        print("overall %.3f"%np.mean(tot), {t:round(np.mean(v),3) for t,v in accs.items()}, "used", np.mean(used), "exhaust", ex)
    return np.mean(tot)
if __name__=="__main__":
    run(reps=int(sys.argv[1]) if len(sys.argv)>1 else 5)
