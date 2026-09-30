import json, numpy as np, csv
from math import pi, sin, cos
# ---- assemble data: (N, D, sched, f, kind, loss). kind: 'ckpt' or 'final' (wsd branch final => final with D=f*D)
rows=[]
nb=json.load(open('notebook/checkpoints.json'))
nbr={int(r['run']):r for r in csv.DictReader(open('notebook/runs.csv'))}
for e in nb:
    r=nbr[e['run']]; N=float(r['N']); D=float(r['D']); s=r['sched']
    for c in e['checkpoints']: rows.append((N,D,s,c['frac'],'ckpt',c['loss']))
    rows.append((N,D,s,1.0,'ckpt',float(r['loss'])))
for line in open('lab_runs.jsonl'):
    j=json.loads(line); c=j['config']
    if j['status']!='ok': continue
    N=c['N'];D=c['D'];s=c['sched']
    for k in j.get('checkpoints') or []: rows.append((N,D,s,k['frac'],'ckpt',k['loss']))
    if 1.0 not in [k['frac'] for k in (j.get('checkpoints') or [])]: rows.append((N,D,s,1.0,'ckpt',j['loss']))
    for k in j.get('cooldown_branches') or []: rows.append((N,D*k['frac'],'wsd',1.0,'ckpt',k['loss']))
print(len(rows),'points')
def lr(s,f):
    if s=='cosine': return 0.1+0.45*(1+cos(pi*f))
    return 1.0 if f<=0.8 else (1-f)/0.2
def area(s,f):
    if s=='cosine': return 0.1*f+0.45*(f+sin(pi*f)/pi)
    return f if f<=0.8 else 0.8+ (f-0.8) - (f-0.8)**2/0.4
np.save('rows.npy',np.array(rows,dtype=object),allow_pickle=True)
