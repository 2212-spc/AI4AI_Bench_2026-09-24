import csv,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path('/app')
def load():
 rows=list(csv.DictReader(open(ROOT/'data/pool.csv')))
 reviews={m:{int(r['item']):int(r['review_verdict']) for r in csv.DictReader(open(ROOT/f'data/review_{m}_00{j}.csv'))} for j,m in enumerate('AB')}
 records=[]
 for m in 'AB':
  for r in rows:
   i=int(r['item']);j=int(r['judge_'+m]);v=reviews[m][i];l=int(r['len_'+m]);s=f'{j}{v}'
   if m=='B' and s=='11':s+='_'+('short' if l<500 else 'medium' if l<900 else 'long')
   records.append(dict(item=i,model=m,category=r['category'],length=l,judge=j,review=v,stratum=s))
 return records

def labels():
 out={}
 for p in sorted((ROOT/'data').glob('verify_*.csv')):
  for r in csv.DictReader(open(p)):
   out[(r['model'],int(r['item']))]=int(r['verified_correct'])
 return out

if __name__=='__main__':
 records=load();rng=np.random.default_rng(42019)
 alloc={'A':{'11':15,'00':7,'01':4,'10':4},'B':{'11_short':12,'11_medium':14,'11_long':12,'00':4,'01':4,'10':4}}
 sampled=[]
 for m,groups in alloc.items():
  for s,n in groups.items():
   pop=[r for r in records if r['model']==m and r['stratum']==s]
   selected=rng.choice(len(pop),n,replace=False)
   sampled.extend(dict(pop[int(k)],stage=1,population=len(pop),sample_size=n) for k in selected)
 with open(ROOT/'analysis/stage1.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=sampled[0].keys());w.writeheader();w.writerows(sampled)
 for m in 'AB':
  ids=','.join(str(r['item']) for r in sampled if r['model']==m)
  result=subprocess.run(['lab','verify',f'model={m}',f'items={ids}'],capture_output=True,text=True,check=True)
  print(result.stdout)
