import csv,json,math,numpy as np
# Fit the power-law edge in log-space using the observed off-qk, 2% warmup
# stable/diverged constraints. Midpoint of the feasible region is an estimate,
# not an epistemic interval: the law is measurable, unlike the censored Q.
runs=list(csv.DictReader(open('/app/notebook/runs.csv')))
for l in open('/app/lab_runs.jsonl'):
    r=json.loads(l)
    if 'config' in r:
        runs.append(dict(r['config'],status=r['status']))
print('Rows:',len(runs))
rng=np.random.default_rng(917)
a=rng.uniform(math.log(.0033),math.log(.0035),1000000)
delta=rng.uniform(.35,.60,len(a))
mask=np.ones(len(a),bool)
for r in runs:
    if float(r['qk'])!=0 or float(r['wu'])!=.02:continue
    pred=a-delta*np.log(float(r['N'])/3e8)
    if r['status']=='ok':mask &= pred>=np.log(float(r['lr']))
    else:mask &= pred<=np.log(float(r['lr']))
a=a[mask]; delta=delta[mask]
prod=a-delta*np.log(7e9/3e8)
print('delta quantiles',np.quantile(delta,[0,.5,1]))
print('production log10 edge quantiles',np.quantile(prod/np.log(10),[0,.5,1]))
print('production edge estimate',np.exp(np.median(prod)))
print('N=3e8 edge estimate',np.exp(np.median(a)))
