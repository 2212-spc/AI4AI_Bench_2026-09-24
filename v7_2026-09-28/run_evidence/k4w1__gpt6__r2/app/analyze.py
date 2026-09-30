import json,collections,numpy as np
T=['algebra','geometry','combinatorics','number_theory']
for t in T:
 qs=json.load(open('/app/'+t+'.json'))['questions'];print('\n',t)
 for v in range(5):
  ss=[s for q in qs for s in q['samples'] if s['variant']==v]
  print(v,round(np.mean([s['correct'] for s in ss]),3),end='; ')
 print()
 for c in [True,False]:
  ss=[s['score'] for q in qs for s in q['samples'] if s['correct']==c];print(c,len(ss),np.round(np.percentile(ss,[0,10,25,50,75,90,100]),2))
 for q in qs:
  wrong=collections.Counter(s['answer'] for s in q['samples'] if not s['correct']);trap=wrong.most_common(1)[0][0] if wrong else ''
  counts=[sum(s['correct'] for s in q['samples'] if s['variant']==v) for v in range(5)]
  big=[(a,n,sorted(set(s['variant'] for s in q['samples'] if s['answer']==a)),round(np.mean([s['score'] for s in q['samples'] if s['answer']==a]),2)) for a,n in wrong.most_common(5) if n>=2]
  print(counts,'correct score',round(np.mean([s['score'] for s in q['samples'] if s['correct']]),2), 'wrong',big)
