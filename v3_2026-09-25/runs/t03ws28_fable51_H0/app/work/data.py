import json, csv, numpy as np
rows=[]
for l in open('/app/lab_runs.jsonl'):
    r=json.loads(l); c=r['config']
    if r['status']=='ok': rows.append((c['N'],c['D'],c['q'],c['sub'],c['seed'],r['loss']))
for r in csv.DictReader(open('/app/notebook/runs.csv')):
    rows.append((float(r['N']),float(r['D']),float(r['q']),float(r['sub']),int(r['seed']),float(r['loss'])))
data=np.array(rows)
