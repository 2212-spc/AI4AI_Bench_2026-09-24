import json, glob, collections
import numpy as np
data = {}
for f in glob.glob('data/dev_*.json'):
    d = json.load(open(f))
    for q in d['questions']:
        data.setdefault(q['topic'], []).append(q)
for t in data:
    sc=[s['score'] for q in data[t] for s in q['samples'] if s['correct']]
    hw=[s['score'] for q in data[t] for s in q['samples'] if not s['correct'] and s['score']>0.2]
    lw=[s['score'] for q in data[t] for s in q['samples'] if not s['correct'] and s['score']<=0.2]
    print(f"{t:15s} correct: n={len(sc)} min={min(sc):.2f} mean={np.mean(sc):.2f} max={max(sc):.2f} | trap-wrong: n={len(hw)} mean={np.mean(hw) if hw else 0:.2f} min={min(hw) if hw else 0:.2f} | noise-wrong: n={len(lw)} max={max(lw):.2f}")
    print("   correct score hist:", np.histogram(sc, bins=[-2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,3])[0])
    if hw: print("   trap score hist:   ", np.histogram(hw, bins=[-2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,3])[0])
# Trap structure: for each (question, variant), is there a trap answer? How many distinct traps per question, shared across how many variants
print("\n=== per-question trap analysis (geometry+algebra) ===")
for t in ['geometry','algebra']:
    for q in data[t]:
        traps=collections.defaultdict(set)
        for s in q['samples']:
            if not s['correct'] and s['score']>0.2: traps[s['answer']].add(s['variant'])
        cv=set(s['variant'] for s in q['samples'] if s['correct'])
        print(f"{t[:4]} {q['id']} correct in variants {sorted(cv)} ; traps: " + ", ".join(f"{a}->v{sorted(v)}" for a,v in traps.items()))
