import json, subprocess, sys
def train(n, steps, eta, seed=None):
    args = [f"n={n}", f"steps={steps}", f"eta={eta}"]
    if seed is not None: args.append(f"seed={seed}")
    out = subprocess.run(["/app/bin/lab", "train"] + args, capture_output=True, text=True).stdout
    r = json.loads(out)
    return r
if __name__ == "__main__":
    n=int(sys.argv[1]); steps=int(sys.argv[2])
    for eta in sys.argv[3].split(","):
        seed = int(sys.argv[4]) if len(sys.argv)>4 else None
        r = train(n, steps, float(eta), seed)
        print(f"n={n} steps={steps} eta={float(eta):.3e} div={r['diverged']} loss={r['final_loss']} at={r.get('diverged_at_step')} left={r['budget_left']:.3e}")
