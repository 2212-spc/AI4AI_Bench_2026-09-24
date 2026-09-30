import json, os, sys, urllib.request
URL = os.environ.get("LAB_URL", "http://127.0.0.1:18787") + "/api"
TOK = open("/app/.lab_token").read().strip()
CACHE = "/app/work/runs.jsonl"

def call(op, args):
    body = json.dumps({"token": TOK, "op": op, "args": args}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=170) as r:
        return json.loads(r.read())

def run(batch, tokens, lr, wd, seeds=1):
    out = call("run", dict(batch=batch, tokens=tokens, lr=lr, wd=wd, seeds=seeds))
    if not out.get("ok"):
        raise RuntimeError(out)
    r = out["result"]
    with open(CACHE, "a") as f:
        f.write(json.dumps(r) + "\n")
    return r

def load():
    if not os.path.exists(CACHE): return []
    return [json.loads(l) for l in open(CACHE)]

if __name__ == "__main__":
    b, t, lr, wd = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    s = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    r = run(b, t, lr, wd, s)
    print(f"B={b} T={t} lr={lr:.3e} wd={wd:.4f} steps={r['steps']} loss={r['final_val_loss']:.4f} per_seed={r['per_seed']} left={r['budget_left']}")
