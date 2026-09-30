"""Generator for family E1: budgeted recipe-change triage with interacting changes.

Truth  = 64-cell full factorial x many seeds (common random numbers), computed by execution.
Certificates (all by execution / simulation against the truth table):
  C1 truth resolution : accept boundary delta sits in a gap of the sorted regrets (no cell within 2 paired SE)
  C2 structure        : OFAT-combine set is outside the accept set; greedy forward ends outside it
  C3 solvability      : a blind expert policy (factorial screen + replication, <= budget) passes >= 0.8
  C4 no shortcut      : ship-all, ship-none, OFAT(-verify), forward greedy, random each pass <= 0.2
                        (backward elimination is reported as a competent strategy, not required to fail)
  C5 estimation       : predicted-MSE tolerance is >= 2.5x the per-run noise of a 5-seed mean
usage: gen_e1.py <world.json> <truth.npz> <instance_id>"""
import json, os, shutil, sys
sys.path.insert(0, "/tmp/bench/gen"); sys.path.insert(0, "/tmp/bench/lab")
import numpy as np
import e1_world as W
import engine as E

BUDGET = int(os.environ.get("E1_BUDGET", 80))
PRED_TOL = 0.06
N_SIM = 600


def load(truth):
    z = np.load(truth); return z["V"], z["D"]


def pick_delta(mu, V):
    opt = int(np.argmin(mu)); reg = mu / mu[opt] - 1
    se_pair = np.array([np.std(V[m] - V[opt], ddof=1) / np.sqrt(V.shape[1]) for m in range(64)]) / mu[opt]
    best = None
    for d in np.arange(0.020, 0.0451, 0.0005):
        clear = np.min(np.abs(reg - d) / np.maximum(se_pair, 1e-9))
        if best is None or clear > best[1]: best = (float(d), float(clear))
    return opt, reg, best


# ---------------- policy simulations (each returns (mask, predicted, runs_used)) ----------------
def _screen_then_replicate(V, rng, cells, n1):
    """run every cell in `cells` on n1 common seeds, then spend the rest of the budget replicating the top cells."""
    rest = BUDGET - len(cells) * n1
    if rest < 0: return 0, float(V[0].mean()), 0
    k = 4 if rest >= 16 else 2; r = rest // k
    S = V.shape[1]; seeds = rng.choice(S, n1 + r, replace=False)
    obs = V[cells][:, seeds[:n1]].mean(1); top = cells[np.argsort(obs)[:k]]
    means = {m: V[m, seeds].mean() for m in top}
    m = min(means, key=means.get); return int(m), float(means[m]), len(cells) * n1 + k * r


def P_expert_full(V, rng):
    return _screen_then_replicate(V, rng, np.arange(64), 1)


def P_expert_prune(V, rng):
    """domain knowledge: c6 only re-draws the minibatch RNG stream, so its expected effect is zero -> drop it."""
    return _screen_then_replicate(V, rng, np.arange(32), 2 if BUDGET >= 80 else 1)


def P_ofat(V, rng, r=None):
    r = r or BUDGET // 7
    S = V.shape[1]; seeds = rng.choice(S, r, replace=False)
    base = V[0, seeds]; m = 0; pred = base.mean()
    for i in range(6):
        d = (V[1 << i, seeds] - base).mean()
        if d < 0: m |= 1 << i; pred += d
    return m, float(pred), 7 * r


def P_ofat_verify(V, rng, r=None):
    r = r or BUDGET // 8
    S = V.shape[1]; seeds = rng.choice(S, r, replace=False)
    base = V[0, seeds]; m = 0; singles = {}
    for i in range(6):
        singles[1 << i] = V[1 << i, seeds].mean()
        if singles[1 << i] < base.mean(): m |= 1 << i
    combo = V[m, seeds].mean(); bs = min(singles, key=singles.get)
    if combo <= singles[bs]: return m, float(combo), 8 * r
    return int(bs), float(singles[bs]), 8 * r


def P_greedy_fwd(V, rng, r=None):
    r = r or max(2, BUDGET // 20)
    S = V.shape[1]; seeds = rng.choice(S, r, replace=False); used = r
    cur = 0; cv = V[0, seeds].mean()
    while True:
        cands = [cur | 1 << i for i in range(6) if not cur >> i & 1]
        if not cands or used + r * len(cands) > BUDGET: break
        vals = {c: V[c, seeds].mean() for c in cands}; used += r * len(cands)
        c = min(vals, key=vals.get)
        if vals[c] >= cv: break
        cur, cv = c, vals[c]
    return cur, float(cv), used


def P_greedy_bwd(V, rng, r=None):
    r = r or max(2, BUDGET // 20)
    S = V.shape[1]; seeds = rng.choice(S, r, replace=False); used = r
    cur = 63; cv = V[63, seeds].mean()
    while True:
        cands = [cur & ~(1 << i) for i in range(6) if cur >> i & 1]
        if not cands or used + r * len(cands) > BUDGET: break
        vals = {c: V[c, seeds].mean() for c in cands}; used += r * len(cands)
        c = min(vals, key=vals.get)
        if vals[c] >= cv: break
        cur, cv = c, vals[c]
    return cur, float(cv), used


def P_all(V, rng):
    return 63, float(V[63].mean()), 0


def P_none(V, rng):
    return 0, float(V[0].mean()), 0


def P_random(V, rng):
    m = int(rng.integers(0, 64)); return m, float(V[m].mean()), 0


POLICIES = {"expert_full_factorial": P_expert_full, "expert_prune_c6": P_expert_prune, "ofat_combine": P_ofat,
            "ofat_then_verify": P_ofat_verify, "greedy_forward": P_greedy_fwd, "greedy_backward": P_greedy_bwd,
            "ship_all": P_all, "ship_none": P_none, "random_subset": P_random}
EXPERT = ["expert_full_factorial", "expert_prune_c6"]
COMPETENT = ["greedy_backward"]  # evaluates every change in the context of the others: a legitimate strategy, reported not required to fail


def score(mu, opt, delta, mask, pred):
    ok_dec = mu[mask] <= mu[opt] * (1 + delta)
    ok_pred = pred is not None and abs(pred - mu[mask]) / mu[mask] <= PRED_TOL
    return bool(ok_dec), bool(ok_pred)


def simulate(V, mu, opt, delta, seed=0):
    rng = np.random.default_rng(seed); out = {}
    for name, f in POLICIES.items():
        n_dec = n_both = 0; ends = {}
        for _ in range(N_SIM):
            m, pred, used = f(V, rng); assert used <= BUDGET
            a, b = score(mu, opt, delta, m, pred); n_dec += a; n_both += a and b
            k = "+".join(W.names_of(m)) or "(none)"; ends[k] = ends.get(k, 0) + 1
        top = sorted(ends.items(), key=lambda t: -t[1])[:3]
        out[name] = {"pass_decision": n_dec / N_SIM, "pass": n_both / N_SIM, "typical_end": [(k, round(c / N_SIM, 2)) for k, c in top]}
    return out


# ---------------- public materials ----------------
CHANGES = {
    "c1": "Raise the peak learning rate from {lr} to {lr2}.",
    "c2": "Raise SGD momentum from {mom} to {mom2}.",
    "c3": "Enable global gradient-norm clipping (max norm {clip}).",
    "c4": "Widen the hidden layer from 96 to {hid2} units (same initialisation scheme).",
    "c5": "Add L2 weight decay {wd2} (coupled, added to the gradient as in torch.optim.SGD).",
    "c6": "Draw minibatch indices from a dedicated RNG stream instead of the RNG used for initialisation "
          "(decouples data order from init).",
}


def teammate_seed(world, acc=None):
    """Honest single-seed OFAT note. Legacy wA condition first (keeps e1-wA reproducible); otherwise the first seed
    whose 'ship the single-seed winners' plan lies outside the accept set (the note is a real run, never fabricated)."""
    cache = []
    for s in range(200):
        r = {m: W.run_cell(world, m, s)[0] for m in [0, 1, 2, 4, 8, 16, 32]}
        b = r[0]; rel = {m: r[m] / b - 1 for m in r}; cache.append((s, r, rel))
        if rel[1] < -0.05 and rel[2] < -0.05 and rel[8] < -0.03 and rel[32] < -0.01 and rel[4] > 0.005 and abs(rel[16]) < 0.01:
            return s, r
        if s >= 30: break
    for s, r, rel in cache:
        plan = sum(m for m in rel if m and rel[m] < -0.005)
        if plan and (acc is None or plan not in acc): return s, r
    raise RuntimeError("no teammate seed")


def render_public(pub, params, world, vtriv, acc=None):
    os.makedirs(pub + "/notes", exist_ok=True)
    fmt = dict(params, lr2=round(params["lr"] * params["lr_f"], 4), hid2=params.get("hid2", 160), wd2=params.get("wd2", 3e-4))
    b = world["base"]
    open(pub + "/RECIPE.md", "w").write(f"""# MiniLab production recipe (current)

Model: 2-layer MLP (d_in=32 -> hidden {b['hidden']} ReLU -> 1), He-normal first layer, output layer std 1/sqrt(hidden), zero biases.
Task: regression, mean-squared error. Training set 20,000 examples, fixed validation set 4,000 examples (both fixed; only the
training seed varies between runs).
Optimiser: SGD with momentum {b['momentum']} (dampening 0), peak lr {b['lr']}, linear warm-up {b['warmup_steps']} steps then
cosine decay to {b['min_lr_ratio']} x peak, {b['steps']} steps, batch size {b['micro_batch_size']}, no weight decay,
no gradient clipping.
The training seed controls initialisation and minibatch order.

Metric: validation MSE of the final weights (lower is better). The quantity that matters for shipping is the
**expected** validation MSE over training seeds. A run that diverges produces a useless model and is counted as
val MSE = {vtriv:.4f} (the MSE of predicting the mean).
""")
    open(pub + "/CHANGES.md", "w").write("# Proposed recipe changes\n\nEach change is a small, independent edit to the recipe; any subset can be combined.\n\n" +
                                          "\n".join(f"- **{k}**: {v.format(**fmt)}" for k, v in CHANGES.items()) + "\n")
    s, r = teammate_seed(world, acc)
    rel = {m: r[m] / r[0] - 1 for m in r}
    win = [W.names_of(m)[0] for m in r if m and rel[m] < -0.005]; hurt = [W.names_of(m)[0] for m in r if m and rel[m] > 0.005]
    flat = [W.names_of(m)[0] for m in r if m and abs(rel[m]) <= 0.005]
    plan = 'Plan: ship the winners together (%s)' % ', '.join(win)
    if hurt: plan += '; drop %s (hurts)' % ', '.join(hurt)
    if flat: plan += (' and ' if hurt else '; drop ') + '%s (no effect)' % ', '.join(flat)
    plan += '. Didn\'t have time to run the combination.\n'
    lines = [f"| {'baseline' if m == 0 else W.names_of(m)[0]} | {r[m]:.4f} | {'' if m == 0 else '%+.1f%%' % (100 * (r[m] / r[0] - 1))} |" for m in r]
    open(pub + "/notes/teammate_ablation.md", "w").write(
        "# Ablation notes (teammate, last week)\n\nI tried each proposed change on its own against the baseline (one run each, seed %d).\n\n"
        "| run | val MSE | vs baseline |\n|---|---|---|\n" % s + "\n".join(lines) +
        "\n\n" + plan)
    return s


INSTR = """# Recipe change triage under a compute budget

You own the MiniLab training recipe. `/app/RECIPE.md` describes the current production recipe, `/app/CHANGES.md`
lists six proposed changes (c1-c6) and `/app/notes/teammate_ablation.md` has a teammate's quick ablation and plan.

Decide which subset of the six changes to ship so that the **expected validation MSE (over training seeds) of the
shipped recipe is as low as possible**.

You can train the recipe on the shared cluster with `/app/lab` (run `/app/lab --help`). Every `lab run` costs one
run; the budget for this task is **{budget} runs** in total and further runs are refused. `lab status` and
`lab history` are free. There is no other way to train the real model; the training data are not available locally.

Deliverables:
1. `/app/decision.json`: `{{"ship": ["c.."], "predicted_val_mse": <float>}}` where `predicted_val_mse` is your
   estimate of the expected validation MSE (over seeds) of the recipe with exactly the shipped changes.
2. `/app/REPORT.md`: a short justification with the evidence you relied on.

Grading (hidden truth = many-seed evaluation of every subset): you pass if the true expected val MSE of your
shipped subset is within {delta_pct:.1f}% of the best of all 64 subsets **and** your `predicted_val_mse` is within
{tol_pct:.0f}% of the true expected val MSE of the subset you shipped.
"""
HINT = """
Note from the reviewer: in this recipe the changes interact strongly - some pairs that each help alone make
training unstable when combined, and some changes only help in combination with others. Single-change ablations
and greedy one-at-a-time selection are known to give the wrong answer here. Compare arms on common seeds and
spend budget on combinations.
"""


def main():
    wj, truth, iid = sys.argv[1:4]
    params = json.load(open(wj)); V, D = load(truth); mu = V.mean(1)
    world = W.build_world(params); vtriv = float(np.var(world["data"][3]))
    opt, reg, (delta, clear) = pick_delta(mu, V)
    acc = [m for m in range(64) if reg[m] <= delta]
    cert = {"instance": iid, "params": params, "n_truth_seeds": int(V.shape[1]), "opt": W.names_of(opt), "opt_mu": float(mu[opt]),
            "delta_rel": delta, "boundary_clearance_pairedSE": clear, "accept_set": ["+".join(W.names_of(m)) or "(none)" for m in acc],
            "diverge_rate_by_cell": {"+".join(W.names_of(m)) or "(none)": float(D[m].mean()) for m in range(64) if D[m].any()}}
    sim = simulate(V, mu, opt, delta); cert["policies"] = sim
    per_run_sd = float(np.median(V.std(1, ddof=1) / mu))
    cert["C1_truth_resolution"] = bool(clear >= 2.0)
    ofat_set = sum(1 << i for i in range(6) if mu[1 << i] < mu[0])
    cert["C2_structure"] = bool(ofat_set not in acc and sim["greedy_forward"]["pass_decision"] <= 0.2)
    cert["C3_solvable"] = bool(max(sim[p]["pass"] for p in EXPERT) >= 0.8)
    cert["C4_no_shortcut"] = bool(all(sim[p]["pass_decision"] <= 0.2 for p in sim if p not in EXPERT + COMPETENT))
    cert["C5_estimation"] = bool(PRED_TOL >= 2.5 * per_run_sd / np.sqrt(5))
    cert["accepted"] = bool(all(cert[k] for k in cert if k.startswith("C")))
    root = f"/tmp/bench/instances/e1/{iid}"
    if os.path.exists(root): shutil.rmtree(root)
    os.makedirs(root + "/hidden"); pub = root + "/public"; os.makedirs(pub)
    tseed = render_public(pub, params, world, vtriv, acc); cert["teammate_seed"] = tseed
    shutil.copy("/tmp/bench/lab/lab_cli.py", pub + "/lab"); os.chmod(pub + "/lab", 0o755)
    fmt = dict(budget=BUDGET, delta_pct=100 * delta, tol_pct=100 * PRED_TOL)
    open(root + "/instruction.md", "w").write(INSTR.format(**fmt))
    open(root + "/instruction_hint.md", "w").write(INSTR.format(**fmt) + HINT)
    json.dump(params, open(root + "/hidden/world.json", "w"))
    json.dump({"mu": mu.tolist(), "se": (V.std(1, ddof=1) / np.sqrt(V.shape[1])).tolist(), "opt": opt, "delta_rel": delta,
               "pred_tol": PRED_TOL, "budget": BUDGET}, open(root + "/hidden/scoring.json", "w"))
    json.dump(cert, open(root + "/hidden/certificate.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in cert.items() if k not in ("diverge_rate_by_cell",)}, indent=1))


if __name__ == "__main__":
    main()
