import csv,subprocess
from pathlib import Path
import numpy as np
ROOT=Path('/app')
r=list(csv.DictReader(open(ROOT/'data/pool.csv')))
reviews={m:{int(x['item']):int(x['review_verdict']) for x in csv.DictReader(open(ROOT/f'data/review_{m}_00{0 if m=="A" else 1}.csv'))} for m in 'AB'}
# Existing full-verification items.
seen=set()
for p in (ROOT/'data').glob('verify_*.csv'):
    for x in csv.DictReader(open(p)):
        seen.add((x['model'],int(x['item'])))

def strata(m, i):
    x=r[i]; j=int(x['judge_'+m]); q=reviews[m][i]
    if m=='B' and j==1 and q==1:
        l=int(x['len_B']); return '11_short' if l<500 else '11_medium' if l<900 else '11_long'
    return f'{j}{q}'
alloc={'A':{'00':8,'01':4,'10':4,'11':14},'B':{'00':10,'01':6,'10':10,'11_short':10,'11_medium':12,'11_long':32}}
rng=np.random.default_rng(91931)
for m,groups in {'B':alloc['B']}.items():
  picked=[]
  for s,n in groups.items():
    pop=[i for i in range(600) if (m,i) not in seen and strata(m,i)==s]
    if len(pop)<n: raise RuntimeError((m,s,len(pop),n))
    picked += list(rng.choice(pop,n,replace=False))
  ids=','.join(map(str,picked))
  out=subprocess.run(['lab','verify',f'model={m}',f'items={ids}'],capture_output=True,text=True,check=True)
  print(out.stdout)
  with open(ROOT/('analysis/stage2_'+m+'.csv'),'w') as f:
    w=csv.writer(f);w.writerow(['model','item','stratum']);w.writerows((m,i,strata(m,i)) for i in picked)
