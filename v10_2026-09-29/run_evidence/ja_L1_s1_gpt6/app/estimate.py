from audit import *
d=json.load(open(ROOT/'data/final_design.json'))
v={m:{} for m in ['A','B']}
for m,calls in [('A',[2,4]),('B',[3,5])]:
 for c in calls:
  v[m].update({int(r['item']):int(r['verified_correct']) for r in csv.DictReader(open(ROOT/f'data/verify_{m}_{c:03d}.csv'))})
rng=np.random.default_rng(271828)
est={}; draws={};summary=[]
for m in ['A','B']:
 estimate=0;posterior=np.zeros(200000)
 for h,ids in sorted(d['strata'].items()):
  if h[0]!=m:continue
  ss=d['sample'][h];pilot=[i for i in ids if i in v[m] and i not in ss]
  R=len(ids)-len(pilot);n=len(ss);k=sum(v[m][i] for i in ss)
  contribution=sum(v[m][i] for i in pilot)+R*k/n
  estimate+=contribution/600
  observed=[v[m][i] for i in ids if i in v[m]]
  y=sum(observed);unseen=len(ids)-len(observed)
  p=rng.beta(y+.5,len(observed)-y+.5,200000)
  posterior+=y+rng.binomial(unseen,p)
  summary.append({'stratum':h,'population':len(ids),'pilot_n':len(pilot),'main_n':n,'main_passes':k,'verified_n':len(observed),'verified_passes':y,'estimated_passes':contribution})
 est[m]=estimate;draws[m]=posterior/600
out={'estimates':{'A':est['A'],'B':est['B'],'difference':est['B']-est['A'],'decision':'SHIP' if est['B']>est['A'] else 'HOLD'},'strata':summary,'uncertainty':{name:{'mean':float(x.mean()),'interval_95':list(np.quantile(x,[.025,.975]))} for name,x in [('A',draws['A']),('B',draws['B']),('difference',draws['B']-draws['A'])]},'posterior_probability_B_better':float(np.mean(draws['B']>draws['A'])),'credits_spent':500,'verified_A':len(v['A']),'verified_B':len(v['B'])}
json.dump(out,open(ROOT/'data/estimates.json','w'),indent=2)
(ROOT/'answers.md').write_text(f"Q1: {est['B']:.3f}\nQ2: {est['B']-est['A']:+.3f}\nQ3: {out['estimates']['decision']}\n")
print(json.dumps({k:value for k,value in out.items() if k!='strata'},indent=2))
