exec(open('/app/analyze.py').read().split('for t in sorted')[0])
for t in sorted(set(q['topic'] for q in qs)):
 print('\n',t)
 for q in [q for q in qs if q['topic']==t][:25]:
  print(q['id'], ' '.join('v%d:%s'%(v,','.join(('C' if s['correct'] else ('H'+s['answer'] if s['score']>.6 else 'x')) for s in q['samples'] if s['variant']==v)) for v in range(5)))
