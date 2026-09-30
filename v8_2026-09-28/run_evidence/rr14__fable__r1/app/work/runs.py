#!/usr/bin/env python3
"""Run a batch of lab configs concurrently. Usage: runs.py tokens seeds lr1:wd1 lr2:wd2 ..."""
import sys, json, os
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/app/bin')
import importlib.util
from importlib.machinery import SourceFileLoader
lab = SourceFileLoader("lab", "/app/bin/lab").load_module()

tokens = float(sys.argv[1]); seeds = int(sys.argv[2])
cfgs = [tuple(map(float, a.split(':'))) for a in sys.argv[3:]]
def go(c):
    lr, wd = c
    args = {"batch": 512, "tokens": tokens, "lr": lr, "wd": wd, "seeds": seeds}
    out = lab.call("run", args)
    if not out.get("ok"):
        return {"args": args, "error": out}
    with open("/app/lab_log.jsonl", "a") as f:
        f.write(json.dumps({"op": "run", "args": args, "result": out["result"]}) + "\n")
    with open("/app/work/results.jsonl", "a") as f:
        f.write(json.dumps({"args": args, "result": out["result"]}) + "\n")
    return {"args": args, "result": out["result"]}
with ThreadPoolExecutor(8) as ex:
    for r in ex.map(go, cfgs):
        a = r["args"]
        if "error" in r: print(a, r["error"]); continue
        res = r["result"]
        print(f"T={a['tokens']:>6} lr={a['lr']:.5f} wd={a['wd']:.4f}  loss={res['final_val_loss']:.4f}  seeds={res['per_seed']}  left={res['budget_left']}")
