#!/usr/bin/env python3
"""Run a list of (batch,tokens,lr,wd,seeds) jobs concurrently, append to /app/work/results.jsonl."""
import json, sys, os, concurrent.futures as cf
sys.path.insert(0, '/app/bin')
import importlib.util
from importlib.machinery import SourceFileLoader
lab = SourceFileLoader("lab", "/app/bin/lab").load_module()

def one(job):
    out = lab.call('run', job)
    return job, out

jobs = [json.loads(l) for l in open(sys.argv[1])]
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for job, out in ex.map(one, jobs):
        if not out.get('ok'):
            print('ERR', job, out); continue
        r = out['result']
        rec = dict(job); rec['result'] = r
        with open('/app/work/results.jsonl','a') as f: f.write(json.dumps(rec)+'\n')
        print(json.dumps(job), '->', json.dumps(r))
