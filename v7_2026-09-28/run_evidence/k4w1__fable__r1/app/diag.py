import sim, random, importlib.util, collections
qs = sim.load()
print(len(qs), collections.Counter(q['topic'] for q in qs))
spec = importlib.util.spec_from_file_location('pol', '/app/cand3.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
fails = collections.Counter(); tot = collections.Counter()
detail = collections.defaultdict(list)
for seed in range(6):
    env = sim.Env(qs, 6, seed); mod.run(env)
    for q in qs:
        tot[q['topic']] += 1
        if env.submitted[q['id']] != q['correct']:
            fails[q['topic']] += 1
            detail[q['id']].append(env.submitted[q['id']])
print({t: round(1 - fails[t]/tot[t], 3) for t in tot})
# describe the failing questions: what does the full pool look like
for qid, subs in sorted(detail.items(), key=lambda kv: -len(kv[1])):
    q = next(x for x in qs if x['id'] == qid)
    if q['topic'] not in ('algebra', 'geometry'): continue
    g = {}
    for v in range(5):
        for a, s in q['pools'][v]:
            g.setdefault(a, []).append((v, s))
    rows = sorted(g.items(), key=lambda kv: -len(kv[1]))[:5]
    desc = ' | '.join(f"{a}{'*' if a==q['correct'] else ''} n={len(l)} nv={len(set(v for v,_ in l))} m={sum(s for _,s in l)/len(l):.2f}" for a, l in rows)
    print(f"{qid} {q['topic'][:3]} fails={len(subs)}/6 sub={collections.Counter(subs).most_common(2)} :: {desc}")
