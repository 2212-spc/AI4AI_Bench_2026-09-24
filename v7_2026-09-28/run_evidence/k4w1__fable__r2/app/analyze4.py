import json, glob, collections
import numpy as np
data = {}
for f in glob.glob('data/dev*.json'):
    d = json.load(open(f))
    for q in d['questions']:
        data.setdefault(q['topic'], []).append(q)
for t in data:
    rates=sorted(sum(s['correct'] for s in q['samples'])/20 for q in data[t])
    print(t, "n=%d"%len(rates), "per-q correct rate:", np.round(rates,2))
    # per variant correct rate per question
print()
for t in ['geometry','algebra']:
    print("==",t,"per-question per-variant correct counts (of 4) + #traps in that variant")
    for q in data[t]:
        row=[]
        for v in range(5):
            ss=[s for s in q['samples'] if s['variant']==v]
            c=sum(s['correct'] for s in ss)
            tr=collections.Counter(s['answer'] for s in ss if not s['correct'] and s['score']>0.2)
            row.append(f"{c}c/{'+'.join(str(x) for x in tr.values()) or '-'}")
        print(q['id'], " ".join(f"{r:8s}" for r in row))
