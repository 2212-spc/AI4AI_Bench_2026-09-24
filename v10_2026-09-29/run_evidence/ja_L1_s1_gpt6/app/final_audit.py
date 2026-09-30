from audit import *
d=json.load(open(ROOT/'data/design.json'))
verified={m:{} for m in ['A','B']}
for m,call in [('A',2),('B',3)]:
 verified[m]={int(r['item']):int(r['verified_correct']) for r in csv.DictReader(open(ROOT/f'data/verify_{m}_{call:03d}.csv'))}
strata={}
for m in ['A','B']:
 for r in rows:
  i=int(r['item']);j=int(r['judge_'+m]);v=reviews[m][i];l=int(r['len_'+m])
  if m=='A': h=f'A_{j}{v}'
  elif (j,v)==(1,1):h=f'B_11_{int(l>800)+int(l>1100)+int(l>1600)}'
  elif (j,v)==(1,0):h=f'B_10_{int(l>800)}'
  else:h=f'B_{j}{v}'
  strata.setdefault(h,[]).append(i)
counts={'A_00':3,'A_01':1,'A_10':1,'A_11':5,'B_00':6,'B_01':0,'B_10_0':10,'B_10_1':2,'B_11_0':4,'B_11_1':30,'B_11_2':33,'B_11_3':31}
# B_01 has nine unverified cases. Reserve two from the largest B_11 bin
# so every nonempty remaining stratum has positive sampling probability.
counts['B_01']=2;counts['B_11_3']=29
rng=np.random.default_rng(184376)
selected={}
for h,ids in sorted(strata.items()):
 remaining=[i for i in ids if i not in verified[h[0]]]
 n=counts[h]
 assert n<=len(remaining),(h,n,len(remaining))
 selected[h]=sorted(map(int,rng.choice(remaining,n,replace=False)))
 print(h,'N',len(ids),'pilot',len(ids)-len(remaining),'new',n)
assert sum(map(len,selected.values()))==126
json.dump({'strata':strata,'sample':selected,'counts':counts,'seed':184376},open(ROOT/'data/final_design.json','w'),indent=2)
for m in ['A','B']:
 ids=sorted(i for h,ii in selected.items() if h.startswith(m) for i in ii)
 p=subprocess.run(['lab','verify',f'model={m}','items='+','.join(map(str,ids))],capture_output=True,text=True,check=True)
 print(p.stdout)
