import json, os, sys, urllib.request, time
URL = os.environ.get("LAB_URL", "http://127.0.0.1:18787") + "/api"
TOK = open("/app/.lab_token").read().strip()
LOG = "/app/work/runs.jsonl"

def call(op, args):
    body = json.dumps({"token": TOK, "op": op, "args": args}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=170) as r:
        return json.loads(r.read())

def run(batch, tokens, lr, wd, seeds=1):
    args = dict(batch=batch, tokens=tokens, lr=lr, wd=wd, seeds=seeds)
    out = call("run", args)
    if not out.get("ok"):
        print("ERR", args, out); return None
    rec = dict(args); rec["result"] = out["result"]
    with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n")
    return out["result"]

def load():
    if not os.path.exists(LOG): return []
    return [json.loads(l) for l in open(LOG)]

if __name__ == "__main__":
    b, t, lr, wd = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    s = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    print(json.dumps(run(b, t, lr, wd, s)))
