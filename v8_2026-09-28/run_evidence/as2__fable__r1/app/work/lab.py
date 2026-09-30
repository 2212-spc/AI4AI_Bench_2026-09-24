import json, os, urllib.request, itertools
URL = os.environ.get("LAB_URL", "http://127.0.0.1:18787") + "/api"
TOK = open("/app/.lab_token").read().strip()
def call(op, **args):
    body = json.dumps({"token": TOK, "op": op, "args": args}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=170).read())
    if not out.get("ok"): raise RuntimeError(out)
    with open("/app/work/calls.jsonl","a") as f: f.write(json.dumps({"op":op,"args":args,"result":out["result"]})+"\n")
    return out["result"]
IDS = ["C-114","C-127","C-203","C-241","C-318","C-352"]
COST = {"C-114":4,"C-127":3,"C-203":5,"C-241":4,"C-318":4,"C-352":3}
def feasible():
    sets=[()]
    for r in (1,2,3):
        for c in itertools.combinations(IDS,r):
            if sum(COST[x] for x in c)<=11: sets.append(c)
    return sets
