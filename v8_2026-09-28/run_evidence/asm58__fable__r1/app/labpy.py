import json, os, sys, urllib.request
URL = os.environ.get("LAB_URL", "http://127.0.0.1:18787") + "/api"
TOK = open("/app/.lab_token").read().strip()
LOG = "/app/mylog.jsonl"

def call(op, **args):
    body = json.dumps({"token": TOK, "op": op, "args": args}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=170) as r:
        out = json.loads(r.read())
    if not out.get("ok"):
        raise RuntimeError(out)
    with open(LOG, "a") as f:
        f.write(json.dumps({"op": op, "args": args, "result": out["result"]}) + "\n")
    return out["result"]

def replay(opts, trace, reps=1):
    return call("replay", opts=list(opts), trace=trace, reps=reps)

def canary(opts, reps=1):
    return call("canary", opts=list(opts), reps=reps)

def load():
    rows = []
    if os.path.exists(LOG):
        for line in open(LOG):
            rows.append(json.loads(line))
    return rows
