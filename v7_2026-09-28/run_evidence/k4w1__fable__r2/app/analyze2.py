import json, glob, collections
import numpy as np
data = {}
for f in glob.glob('data/dev_*.json'):
    d = json.load(open(f))
    for q in d['questions']:
        data.setdefault(q['topic'], []).append(q)
for t in ['geometry','algebra']:
    print("=====", t)
    for q in data[t][:8]:
        print("Q", q['id'], "correct:", q['correct_answer'])
        for v in range(5):
            ss=[s for s in q['samples'] if s['variant']==v]
            print("  v%d "%v, "  ".join(f"{s['answer']}({s['score']:+.2f}){'*' if s['correct'] else ''}" for s in ss))
# Is the high-score wrong answer the same across variants?
print("\n=== high-score (>0.3) wrong answers: distinct count per question, by variant spread ===")
for t in ['geometry','algebra']:
    nd=[]; per_variant_consistency=[]
    for q in data[t]:
        hw=[s for s in q['samples'] if not s['correct'] and s['score']>0.3]
        answers=collections.Counter(s['answer'] for s in hw)
        nd.append(len(answers))
        # which variants produce the top wrong
        if answers:
            top=answers.most_common(1)[0][0]
            vs=collections.Counter(s['variant'] for s in hw if s['answer']==top)
            per_variant_consistency.append(dict(vs))
    print(t, "distinct high-score wrong answers per q:", nd)
    print(t, "variant spread of top wrong:", per_variant_consistency[:10])
