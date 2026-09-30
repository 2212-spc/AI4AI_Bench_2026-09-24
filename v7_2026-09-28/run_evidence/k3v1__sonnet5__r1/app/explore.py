import subprocess, json, sys

def train(n, steps, eta, seed=None):
    args = f"n={n} steps={steps} eta={eta}"
    if seed is not None:
        args += f" seed={seed}"
    cmd = f"/app/bin/lab train {args}"
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except Exception:
        print("PARSE FAIL", out, file=sys.stderr)
        return None

if __name__ == "__main__":
    import itertools
    results = []
    # small edge scan at n=362, steps=4000
    n = 362
    steps = 4000
    etas = [1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,2e-2,3e-2,5e-2,8e-2]
    for eta in etas:
        r = train(n, steps, eta)
        r['test_n']=n; r['test_steps']=steps; r['test_eta']=eta
        results.append(r)
        print(json.dumps(r))
    with open('/app/scan1.jsonl','w') as f:
        for r in results:
            f.write(json.dumps(r)+"\n")
