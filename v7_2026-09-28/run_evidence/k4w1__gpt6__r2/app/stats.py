exec(open('/app/analyze.py').read().split('for t in T:')[0])
for t in T:
 qs=json.load(open('/app/'+t+'.json'))['questions']; print('\n',t)
 types=collections.defaultdict(list)
 scores=collections.defaultdict(list)
 for q in qs:
  cnt=collections.Counter(s['answer'] for s in q['samples'] if not s['correct'])
  traps=[a for a,n in cnt.items() if n>=2]
  typ='noise' if not traps else ('shared' if any(len(set(s['variant'] for s in q['samples'] if s['answer']==a))>1 for a in traps) else 'local')
  types[typ].append(q)
  for s in q['samples']: scores['correct' if s['correct'] else ('trap' if s['answer'] in traps else 'noise')].append(s['score'])
 for x,ss in scores.items(): print(x,len(ss),np.mean(ss),np.std(ss))
 for typ,qq in types.items():
  print(typ,len(qq),'accuracy',np.mean([s['correct'] for q in qq for s in q['samples']]))
  for q in qq:
   if typ!='shared':continue
   print('  ', [(sum(s['correct'] for s in q['samples'] if s['variant']==v),round(np.mean([s['score'] for s in q['samples'] if s['variant']==v]),1)) for v in range(5)])
