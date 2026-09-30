#!/usr/bin/env python3
import json, itertools
from run_lab import train

UNIQUE = {"web": 3000000000000.0, "code": 36410989526.00656, "math": 9967276924.74475, "papers": 23536988791.987087}
D_TARGET = 5e11

def pool_for(D_proxy):
    s = D_TARGET / D_proxy
    return {k: v / s for k, v in UNIQUE.items()}

MIXES = [
    (0.85,0.05,0.05,0.05),
    (0.70,0.10,0.10,0.10),
    (0.55,0.15,0.15,0.15),
    (0.40,0.20,0.20,0.20),
    (0.25,0.25,0.25,0.25),
    (0.50,0.30,0.10,0.10),
    (0.50,0.10,0.30,0.10),
    (0.50,0.10,0.10,0.30),
    (0.30,0.40,0.20,0.10),
    (0.30,0.20,0.40,0.10),
    (0.30,0.10,0.20,0.40),
    (0.20,0.30,0.30,0.20),
    (0.60,0.20,0.05,0.15),
    (0.60,0.05,0.20,0.15),
    (0.60,0.15,0.15,0.10),
    (0.40,0.30,0.15,0.15),
    (0.40,0.15,0.30,0.15),
    (0.40,0.15,0.15,0.30),
    (0.15,0.35,0.35,0.15),
    (0.10,0.30,0.30,0.30),
    (0.90,0.03,0.03,0.04),
    (0.05,0.60,0.05,0.30),
    (0.05,0.05,0.60,0.30),
    (0.20,0.05,0.50,0.25),
]

N = 5e7
D = 1e9
pool = pool_for(D)

def main():
    results = []
    for m in MIXES:
        mix = dict(zip(["web","code","math","papers"], m))
        assert abs(sum(m) - 1.0) < 1e-9, m
        r = train(N, D, mix, pool)
        res = r["result"]
        row = {"mix": mix, "composite": res["composite"], "eval_loss": res["eval_loss"]}
        results.append(row)
        print(json.dumps(row))
    with open("/app/mix_sweep1.json", "w") as f:
        json.dump(results, f, indent=1)

if __name__ == "__main__":
    main()
