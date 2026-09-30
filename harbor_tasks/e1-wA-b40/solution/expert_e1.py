"""Reference (blind) solution for E1 tasks: factorial screen on common seeds + replication of the leaders.

Uses only what the agent has: /app/RECIPE.md, /app/CHANGES.md and the budgeted `lab` CLI. No hidden files.
Domain reasoning used: c6 only changes which RNG stream draws minibatch indices, so its expected effect on the
seed-averaged MSE is zero -> it is not screened (the screen covers the 32 subsets of c1-c5).
Budget use: 32*n1 screening runs (n1 = 2 if budget >= 80 else 1) + the rest replicating the top-k cells on
fresh common seeds. This is the `expert_prune_c6` policy whose pass rate the generator certifies (C3)."""
import itertools, json, os, re, subprocess, sys

APP = os.environ.get("APP", "/app")
LAB = [sys.executable, os.path.join(APP, "lab")]


def lab(*args):
    out = subprocess.run(LAB + list(args), capture_output=True, text=True)
    d = json.loads(out.stdout)
    if "error" in d: raise RuntimeError(d["error"])
    return d


def main():
    recipe = open(os.path.join(APP, "RECIPE.md")).read()
    vtriv = float(re.search(r"counted as\s+val MSE = ([0-9.]+)", recipe.replace("\n", " ")).group(1))
    st = lab("status"); B = st["remaining"]
    names = ["c1", "c2", "c3", "c4", "c5"]
    cells = [tuple(n for i, n in enumerate(names) if m >> i & 1) for m in range(32)]
    n1 = 2 if B >= 80 else 1
    rest = B - len(cells) * n1
    if rest < 0: raise SystemExit("budget too small for the screen")
    k = 4 if rest >= 16 else 2; r = rest // k
    base = int(os.environ.get("SEED_BASE", "100003"))
    seeds = [base + 7919 * i for i in range(n1 + r)]  # common random numbers across all arms
    obs = {}

    def run(cell, seed):
        d = lab("run", "--changes", ",".join(cell), "--seed", str(seed))
        v = vtriv if d["diverged"] else d["val_mse"]
        obs.setdefault(cell, []).append(v)

    for s in seeds[:n1]:
        for c in cells: run(c, s)
    top = sorted(cells, key=lambda c: sum(obs[c]) / len(obs[c]))[:k]
    for s in seeds[n1:]:
        for c in top: run(c, s)
    best = min(top, key=lambda c: sum(obs[c]) / len(obs[c]))
    pred = sum(obs[best]) / len(obs[best])
    json.dump({"ship": list(best), "predicted_val_mse": pred}, open(os.path.join(APP, "decision.json"), "w"))
    lines = ["# Decision", "", "Ship: %s; predicted expected val MSE %.5f." % ("+".join(best) or "(none)", pred), "",
             "Screen: all 32 subsets of c1-c5 on %d common seed(s); c6 not screened (it only re-draws the minibatch RNG "
             "stream, zero expected effect). Leaders replicated on %d further common seeds:" % (n1, r), ""]
    for c in top:
        lines.append("- %s: mean %.5f over %d runs" % ("+".join(c) or "(none)", sum(obs[c]) / len(obs[c]), len(obs[c])))
    open(os.path.join(APP, "REPORT.md"), "w").write("\n".join(lines) + "\n")
    print(json.dumps({"ship": list(best), "predicted_val_mse": pred}))


if __name__ == "__main__":
    main()
