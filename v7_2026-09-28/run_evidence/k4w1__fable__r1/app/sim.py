import json, random, sys, importlib.util

def load():
    qs = []
    for line in open('/app/lab_log.jsonl'):
        r = json.loads(line)
        if r['op'] != 'dev_sample':
            continue
        for q in r['result']['questions']:
            pools = {v: [] for v in range(5)}
            for s in q['samples']:
                pools[s['variant']].append((s['answer'], s['score']))
            qs.append({'id': q['id'], 'topic': q['topic'], 'correct': q['correct_answer'], 'pools': pools})
    return qs

class Env:
    def __init__(self, qs, budget_per=6, seed=0):
        self.rng = random.Random(seed)
        self.qs = qs
        self.questions = [{'id': q['id'], 'topic': q['topic']} for q in qs]
        self.budget = budget_per * len(qs)
        self.n_variants = 5
        self.topics = ["algebra", "geometry", "combinatorics", "number_theory"]
        self.used = 0
        self.byid = {q['id']: q for q in qs}
        self.pools = {q['id']: {v: list(q['pools'][v]) for v in range(5)} for q in qs}
        for p in self.pools.values():
            for v in p:
                self.rng.shuffle(p[v])
        self.submitted = None
        self.exhausted = 0
    def left(self):
        return self.budget - self.used
    def sample(self, qid, variant=0):
        if self.used >= self.budget:
            raise RuntimeError('pool empty')
        self.used += 1
        p = self.pools[qid][variant]
        if not p:
            # fallback: draw from any variant with remaining (with replacement if all empty)
            self.exhausted += 1
            allp = [x for v in self.pools[qid] for x in self.pools[qid][v]]
            if allp:
                return self.rng.choice(allp)
            q = self.byid[qid]
            allq = [x for v in range(5) for x in q['pools'][v]]
            return self.rng.choice(allq)
        return p.pop()
    def submit(self, out):
        self.submitted = out

def evaluate(policy_path, qs, seeds=range(10), budget_per=6):
    spec = importlib.util.spec_from_file_location('pol', policy_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tot = {}
    cnt = {}
    used = []
    exh = 0
    for seed in seeds:
        env = Env(qs, budget_per, seed)
        mod.run(env)
        used.append(env.used)
        exh += env.exhausted
        for q in qs:
            ok = env.submitted.get(q['id']) == q['correct']
            tot[q['topic']] = tot.get(q['topic'], 0) + ok
            cnt[q['topic']] = cnt.get(q['topic'], 0) + 1
    overall = sum(tot.values()) / sum(cnt.values())
    per = {t: round(tot[t] / cnt[t], 3) for t in sorted(tot)}
    print(f'{policy_path}: overall={overall:.3f} per={per} used={sum(used)/len(used):.0f}/{env.budget} exhausted_draws={exh/len(list(seeds)):.1f}')
    return overall

if __name__ == '__main__':
    qs = load()
    evaluate(sys.argv[1] if len(sys.argv) > 1 else '/app/policy.py', qs)
