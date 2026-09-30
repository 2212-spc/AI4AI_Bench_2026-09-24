#!/usr/bin/env python3
import json, os, sys, urllib.request

URL = os.environ.get("LAB_URL", "http://127.0.0.1:18787") + "/api"
APP = "/app"

def call(op, args):
    tok = open(os.path.join(APP, ".lab_token")).read().strip()
    body = json.dumps({"token": tok, "op": op, "args": args}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=170) as r:
        return json.loads(r.read())

def train(N, D, mix, pool=None):
    args = {"N": N, "D": D, "mix": mix}
    if pool is not None:
        args["pool"] = pool
    return call("train", args)

if __name__ == "__main__":
    print(call(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}))
