import json, sys, collections
import numpy as np
bins=[-3,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,4]
for f in sys.argv[1:]:
    qs=json.load(open(f))['questions']
    c=np.histogram([s['score'] for q in qs for s in q['samples'] if s['correct']],bins)[0]
    w=np.histogram([s['score'] for q in qs for s in q['samples'] if not s['correct']],bins)[0]
    print(f, "acc", round(np.mean([s['correct'] for q in qs for s in q['samples']]),3))
    print(" correct", c); print(" wrong  ", w); print(" P(c|s) ", np.round(c/(c+w+1e-9),2))
    # trap sharing: for each q, traps and number of variants they span
    span=[]; 
    for q in qs:
        traps=collections.defaultdict(set)
        for s in q['samples']:
            if not s['correct'] and s['score']>0.2: traps[s['answer']].add(s['variant'])
        for a,v in traps.items(): span.append(len(v))
    print(" trap variant-span hist:", collections.Counter(span))
    print(" per-q correct rate:", sorted(round(sum(s['correct'] for s in q['samples'])/20,2) for q in qs))
