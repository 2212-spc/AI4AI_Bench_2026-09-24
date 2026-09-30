import json, subprocess, sys
def train(N, D, mix, pool=None):
    args = {"N": N, "D": D, "mix": mix}
    if pool: args["pool"] = pool
    out = subprocess.run(["/app/bin/lab", "train", "--json", json.dumps(args)], capture_output=True, text=True)
    r = json.loads(out.stdout)
    if "eval_loss" not in r: print(r); return None
    return r
if __name__ == "__main__":
    N = float(sys.argv[1]); D = float(sys.argv[2])
    for spec in sys.argv[3:]:
        parts = spec.split("|")
        mix = dict(zip(["web","code","math","papers"], map(float, parts[0].split(","))))
        pool = None
        if len(parts) > 1:
            pool = dict(zip(["web","code","math","papers"], map(float, parts[1].split(","))))
        r = train(N, D, mix, pool)
        if r:
            e = r["eval_loss"]
            print(f"{spec:40s} gen={e['general']:.4f} code={e['code']:.4f} math={e['math']:.4f} comp={r['composite']:.4f} left={r['budget_left']:.3g}")
