import json, glob, collections, statistics as st
import numpy as np
data = {}
for f in glob.glob('data/dev_*.json'):
    d = json.load(open(f))
    for q in d['questions']:
        data.setdefault(q['topic'], []).append(q)
print("=== per-sample accuracy by topic x variant ===")
for t, qs in data.items():
    row = []
    for v in range(5):
        c = [s['correct'] for q in qs for s in q['samples'] if s['variant']==v]
        row.append(f"v{v}:{np.mean(c):.2f}")
    print(f"{t:15s}", " ".join(row))
print("\n=== verifier: mean score correct vs incorrect, AUC-ish by topic ===")
for t, qs in data.items():
    sc = [s['score'] for q in qs for s in q['samples'] if s['correct']]
    si = [s['score'] for q in qs for s in q['samples'] if not s['correct']]
    # AUC
    auc = np.mean([[a>b for b in si] for a in sc]) if sc and si else float('nan')
    print(f"{t:15s} correct {np.mean(sc):.2f}±{np.std(sc):.2f} (n={len(sc)})  wrong {np.mean(si):.2f}±{np.std(si):.2f} (n={len(si)})  AUC={auc:.2f}")
print("\n=== verifier AUC by topic x variant ===")
for t, qs in data.items():
    row=[]
    for v in range(5):
        sc = [s['score'] for q in qs for s in q['samples'] if s['correct'] and s['variant']==v]
        si = [s['score'] for q in qs for s in q['samples'] if not s['correct'] and s['variant']==v]
        auc = np.mean([[a>b for b in si] for a in sc]) if sc and si else float('nan')
        row.append(f"v{v}:{auc:.2f}")
    print(f"{t:15s}", " ".join(row))
print("\n=== wrong answer structure: how often wrong answers repeat (dominant wrong answer share) ===")
for t, qs in data.items():
    shares=[]; correct_any=0; corr_share=[]
    for q in qs:
        ans=[s['answer'] for s in q['samples']]
        cnt=collections.Counter(ans)
        wrong={a:c for a,c in cnt.items() if a!=q['correct_answer']}
        if wrong: shares.append(max(wrong.values())/len(ans))
        corr_share.append(cnt.get(q['correct_answer'],0)/len(ans))
    print(f"{t:15s} max wrong share {np.mean(shares):.2f}  correct share {np.mean(corr_share):.2f}  frac q with correct among 20: {np.mean([c>0 for c in corr_share]):.2f}")
