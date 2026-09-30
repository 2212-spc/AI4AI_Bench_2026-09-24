#!/usr/bin/env python3
import json
from run_lab import train

UNIQUE = {"web": 3000000000000.0, "code": 36410989526.00656, "math": 9967276924.74475, "papers": 23536988791.987087}
D_TARGET = 5e11

def pool_for(D_proxy):
    s = D_TARGET / D_proxy
    return {k: v / s for k, v in UNIQUE.items()}

N = 5e7
D = 1e9
pool = pool_for(D)

combos = []
for code in [0.20, 0.25, 0.30, 0.35]:
    for math in [0.03, 0.05, 0.08, 0.12]:
        for papers in [0.10, 0.15, 0.20]:
            web = 1.0 - code - math - papers
            if web <= 0.1:
                continue
            combos.append((round(web,3), code, math, papers))

print(len(combos), "combos")

def main():
    results = []
    for m in combos:
        mix = dict(zip(["web","code","math","papers"], m))
        r = train(N, D, mix, pool)
        res = r["result"]
        row = {"mix": mix, "composite": res["composite"], "eval_loss": res["eval_loss"]}
        results.append(row)
        print(json.dumps(row))
    with open("/app/mix_sweep2.json", "w") as f:
        json.dump(results, f, indent=1)

if __name__ == "__main__":
    main()
