import json, sys, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL = "http://127.0.0.1:18787/api"
TOK = open('/app/.lab_token').read().strip()
DB = '/app/work/results.jsonl'
def call(op, args):
    body = json.dumps({"token": TOK, "op": op, "args": args}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())
def run(tokens, lr, wd, seeds=1):
    out = call("run", {"batch":256, "tokens":tokens, "lr":lr, "wd":wd, "seeds":seeds})
    rec = {"tokens":tokens, "lr":lr, "wd":wd, "seeds":seeds, "out":out}
    with open(DB, 'a') as f: f.write(json.dumps(rec)+"\n")
    return rec
def many(cfgs, workers=4):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda c: run(*c), cfgs))
def load():
    rows=[]
    for line in open(DB):
        r=json.loads(line)
        if r['out'].get('ok'):
            res=r['out']['result']; r['mean']=res.get('mean', res.get('loss')); r['res']=res
            rows.append(r)
    return rows
if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1])
    for r in many([tuple(c) for c in cfgs]):
        o=r['out']; print(r['tokens'], r['lr'], r['wd'], o.get('result') if o.get('ok') else o)
