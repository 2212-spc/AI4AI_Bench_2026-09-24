import json, glob, collections, random, math
import numpy as np
data = {}
for f in glob.glob('data/dev*.json'):
    d = json.load(open(f))
    for q in d['questions']:
        data.setdefault(q['topic'], []).append(q)

class Pool:
    """Simulated env for a question: draws from the dev samples without replacement per variant."""
    def __init__(self, q, rng):
        self.q=q; self.by_v={v:[s for s in q['samples'] if s['variant']==v] for v in range(5)}
        for v in self.by_v: rng.shuffle(self.by_v[v])
        self.used=0
    def sample(self, v):
        if not self.by_v[v]: raise RuntimeError("exhausted variant %d"%v)
        s=self.by_v[v].pop(); self.used+=1
        return s['answer'], s['score']

TH=0.2
def decide_geo(samples):
    """samples: list of (variant, answer, score). returns answer."""
    cands=collections.defaultdict(list)
    for v,a,s in samples: cands[a].append((v,s))
    best=None; bestkey=None
    for a,lst in cands.items():
        hi=[(v,s) for v,s in lst if s>TH]
        nv=len(set(v for v,s in hi))
        n=len(hi)
        ms=np.mean([s for v,s in hi]) if hi else -9
        # likelihood-ish: correct ~N(1.0,0.4), trap ~N(1.7,0.4)
        llr=sum((( (s-1.7)**2 - (s-1.0)**2 )/(2*0.4**2)) for v,s in hi)
        key=(nv, llr, n, max(s for v,s in lst))
        if bestkey is None or key>bestkey: best, bestkey=a, key
    return best

def policy_geo_rr(pool, budget):
    samples=[]
    for i in range(budget):
        v=i%5
        a,s=pool.sample(v); samples.append((v,a,s))
    return decide_geo(samples)

def policy_geo_adaptive(pool, budget, early=2):
    samples=[]
    for i in range(budget):
        v=i%5
        a,s=pool.sample(v); samples.append((v,a,s))
        if i>=4:
            # early stop: an answer seen in >= early distinct variants with moderate score, and no competitor
            cands=collections.defaultdict(set)
            for vv,aa,ss in samples:
                if ss>TH: cands[aa].add(vv)
            nvs=sorted((len(vs) for vs in cands.values()), reverse=True)
            if nvs and nvs[0]>=early and (len(nvs)==1 or nvs[1]<nvs[0]-1):
                break
    return decide_geo(samples)

def policy_ver(pool, budget, th=0.25):
    best=None; bs=-9
    for i in range(budget):
        a,s=pool.sample(i%5)
        if s>bs: best,bs=a,s
        if bs>th: break
    return best

def evaluate(topic, fn, **kw):
    rng=random.Random(0)
    accs=[]; used=[]
    for rep in range(20):
        for q in data[topic]:
            p=Pool(q, rng)
            ans=fn(p, **kw)
            accs.append(ans==q['correct_answer']); used.append(p.used)
    return np.mean(accs), np.mean(used)

if __name__=="__main__":
    for t in ['combinatorics','number_theory']:
        for b in [6,8,10]:
            print(t, "verifier-stop budget", b, evaluate(t, policy_ver, budget=b))
    for t in ['geometry','algebra']:
        for b in [5,6,8,10,12,15,20]:
            print(t, "rr budget", b, evaluate(t, policy_geo_rr, budget=b))
        for b in [10,15,20]:
            for e in [2,3]:
                print(t, "adaptive budget", b, "early", e, evaluate(t, policy_geo_adaptive, budget=b, early=e))
