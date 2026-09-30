import json, glob, collections
import numpy as np
data = {}
for f in glob.glob('data/dev*.json'):
    d = json.load(open(f))
    for q in d['questions']:
        data.setdefault(q['topic'], []).append(q)
bins=[-3,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,4]
for t in data:
    print("==",t, "n=",len(data[t]))
    print(" per-variant acc:", [round(np.mean([s['correct'] for q in data[t] for s in q['samples'] if s['variant']==v]),3) for v in range(5)])
    print(" per-variant mean correct score:", [round(np.mean([s['score'] for q in data[t] for s in q['samples'] if s['variant']==v and s['correct']]),2) for v in range(5)])
    print(" per-variant mean trap score:", [round(np.mean([s['score'] for q in data[t] for s in q['samples'] if s['variant']==v and not s['correct'] and s['score']>0.2] or [0]),2) for v in range(5)])
    c=np.histogram([s['score'] for q in data[t] for s in q['samples'] if s['correct']],bins)[0]
    w=np.histogram([s['score'] for q in data[t] for s in q['samples'] if not s['correct']],bins)[0]
    print(" bins   ", bins[1:])
    print(" correct", c); print(" wrong  ", w); print(" P(c|s) ", np.round(c/(c+w+1e-9),2))
