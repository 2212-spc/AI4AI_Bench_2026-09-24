"""Instance gates for L1.5 tasks.

  python3 -m l15.gates <task_module> <seed> [--salts 4] [--json out.json]

For one sampled instance: (1) the module's cheap noiseless screen `instance_gate(params)`; (2) every scripted
strategy in `STRATEGIES` is played through a real budgeted Session (same ops, same costs, same noise the agent
gets) under several noise salts, and graded by the same `grade()` the agents get.  Expectations:
  'pass' strategies (oracle = existence proof) must pass on >= 3/4 of salts;
  'fail' strategies (myopic / recitation / nop / cheat) must fail on every salt.
Any violation -> instance rejected."""
import importlib, json, os, sys, tempfile, time
import numpy as np
from .core import Session, rng_for


def run_strategy(mod, params, salt, name, budget=None, truth=None, keep=False):
    bud = budget or mod.BUDGET
    if budget is None and hasattr(mod, "instance_budget"):
        # gate against the same per-instance budget the built task will have, not the module placeholder
        _, _info = mod.instance_gate(params)
        _t = dict(_info)
        if hasattr(mod, "instance_truth"):
            _t.update(mod.instance_truth({"params": params, "salt": salt, "truth": _t}))
        bud = mod.instance_budget(params, _t)
    cfg = {"task": mod.__name__.split(".")[-1], "params": params, "salt": salt, "budget": bud}
    if truth is not None:
        cfg["truth"] = truth
    w = mod.World(cfg)
    art = tempfile.mkdtemp(prefix="gate_art_")
    led = os.path.join(art, "_ledger.jsonl")
    sess = Session(w, led, art)
    fn, expect = mod.STRATEGIES[name]
    err = None
    t0 = time.time()
    try:
        fn(sess, art, rng_for(salt, "strategy", name))
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    g = w.grade(art, sess.records)
    g["strategy"] = name; g["salt"] = salt; g["expect"] = expect; g["error"] = err
    g["spent_frac"] = round(sess.spent / w.budget, 3); g["sec"] = round(time.time() - t0, 1)
    if keep:
        g["art_dir"] = art
    else:
        import shutil; shutil.rmtree(art, ignore_errors=True)
    return g


def _grade_artifact(mod, params, salt, art_dir):
    """Grade an ALREADY-WRITTEN artifact against a fresh instance, with no lab session of its own.

    The ledger is empty on purpose: this asks whether the artifact alone satisfies the rubric on an
    instance it never explored.  Tasks whose grade() inspects the ledger will simply see zero calls.
    """
    bud = mod.BUDGET
    if hasattr(mod, "instance_budget"):
        _, _info = mod.instance_gate(params)
        _t = dict(_info)
        if hasattr(mod, "instance_truth"):
            _t.update(mod.instance_truth({"params": params, "salt": salt, "truth": _t}))
        bud = mod.instance_budget(params, _t)
    w = mod.World({"task": mod.__name__.split(".")[-1], "params": params, "salt": salt, "budget": bud})
    return w.grade(art_dir, [])


def transfer_gate(mod, seeds, n_salts=2, verbose=True):
    """Cross-instance gate: can ONE instance's answer be replayed on the OTHERS?

    `instance_gate` and `gate` both look at a single instance, so neither can see a task whose answer
    barely moves between instances.  That blind spot shipped: k2-mix passed every per-instance gate, and
    then `k2w1__gemflash__r1` scored regret 0.0001 on R1_mixture_quality after four `train` calls that
    were all the SAME uniform mixture at the SAME tiny scale - four repeats of one design point, which
    measure run-to-run noise and cannot identify a mixture.  Replaying that single submission across the
    other gated seeds passed R1 on 4/4 of them (explore/k2a.py); the optimum only moves 0.06-0.08 per
    domain across instances.  The rubric item was free, and no per-instance gate could have told us.

    Here the oracle is run on instance A and its ARTIFACT is graded on instance B.  A rubric item that
    stays satisfied under transfer is not instance-identifying: it rewards recall, not experiment.
    Reported per item rather than as one verdict, because the useful answer is usually "R1 transfers,
    R2 does not" - which localises the defect instead of condemning the whole task.
    """
    import shutil
    items = {}
    for a in seeds:
        pa = mod.sample_params(a)
        if not mod.instance_gate(pa)[0]:
            continue
        src = run_strategy(mod, pa, "xfer-%d" % a, "oracle", keep=True)
        art = src.get("art_dir")
        if not art or not src["pass"]:
            if verbose:
                print("  seed %s: oracle did not pass, skipping as a source" % a)
            continue
        for b in seeds:
            if b == a:
                continue
            pb = mod.sample_params(b)
            if not mod.instance_gate(pb)[0]:
                continue
            for k in range(n_salts):
                g = _grade_artifact(mod, pb, "xfer-%d-%d" % (b, k), art)
                for name, v in g["items"].items():
                    ok = v["ok"] is True or v["ok"] == 1.0
                    items.setdefault(name, [0, 0])
                    items[name][1] += 1
                    items[name][0] += 1 if ok else 0
        shutil.rmtree(art, ignore_errors=True)
    # Transfer means different things for the two kinds of deliverable, and conflating them makes the
    # gate useless.  A .json artifact is a FIXED ANSWER: if it still satisfies a rubric item on an
    # instance it never saw, that item did not require measuring this instance.  A .py artifact is an
    # ALGORITHM that is re-executed against the new instance - it observes that instance through the
    # lab and adapts, so transferring is the POINT, not a leak.  Measured: k6_ramp's online scheduler
    # transfers R1_time 12/12, which is the task working exactly as designed; k2_mix's mixture.json
    # transfers R1 7/9, which is the defect this gate exists to find.
    #
    # So code artifacts are reported but never flagged.  For them the leak to worry about is a
    # different one - hardcoded constants that survive re-execution - and the way to see that is to
    # ablate the constants (explore/k1u.py) rather than to move the artifact.
    code_artifact = any(str(a).endswith(".py") for a in getattr(mod.World, "ARTIFACTS", []))

    # Some rubric items are INTERNAL-CONSISTENCY checks, not claims about the instance: "the file parses",
    # "the fields are present", "the fix moves the knob you yourself named".  Those SHOULD transfer - a
    # well-formed, self-consistent artifact stays well-formed on any instance - so flagging them is pure
    # noise.  Only items that are supposed to encode a MEASUREMENT are held to the transfer standard.
    #
    # Tasks declare these in World.SELF_CONSISTENT; the "R0" prefix is also honoured as a default, but the
    # prefix alone is not enough.  Measured: k8_post's R3_fix_acts_on_cause transfers 8/8 and was flagged
    # as a leak, when in fact it compares the fix to the cause the SAME artifact declares - it can never
    # do anything but transfer.  A naming convention is not a specification.
    #
    # Calibrated against a family known to be sound: on k8_post with seeds 0,1 (cause=capacity) and 6
    # (cause=lr_scale), R1/R2/R3 transfer 2/6 - exactly the two same-cause ordered pairs - while R0
    # transfers 6/6.  Choosing the seeds matters: over 0,1,3 (all cause=capacity) everything transfers
    # 6/6 and the gate looks like it is firing on a sound task.  A transfer gate is only as good as the
    # DIVERSITY of the seeds you hand it, so pass seeds that differ in the instance's headline latent.
    self_consistent = set(getattr(mod.World, "SELF_CONSISTENT", ()))
    exempt = lambda nm: nm in self_consistent or nm.startswith("R0")
    for name, (n_ok, n) in sorted(items.items()):
        if verbose:
            flag = ("<- NOT instance-identifying"
                    if (n and n_ok > 0.5 * n and not exempt(name) and not code_artifact) else "")
            print("  %-34s transfers %d/%d %s" % (name, n_ok, n, flag))
    return {"items": {k: {"transfer_rate": (v[0] / v[1] if v[1] else 0.0), "n": v[1]} for k, v in items.items()},
            "code_artifact": code_artifact,
            "leaky": ([] if code_artifact else
                      [k for k, v in items.items()
                       if v[1] and v[0] > 0.5 * v[1] and not exempt(k)])}


def gate(mod, seed, n_salts=4, verbose=True):
    params = mod.sample_params(seed)
    ok0, info = mod.instance_gate(params)
    res ={"seed": seed, "screen_ok": ok0, "screen": {k: v for k, v in info.items() if not isinstance(v, (list, dict))}, "runs": []}
    if verbose:
        print("seed", seed, "screen", ok0, json.dumps(res["screen"]))
    if not ok0:
        res["accept"] = False
        return res
    bad = []
    for name, (fn, expect) in mod.STRATEGIES.items():
        outs = [run_strategy(mod, params, "gate-%d-%d" % (seed, k), name) for k in range(n_salts if expect == "pass" or name in getattr(mod, "NOISY_FAIL", ()) else min(n_salts, 2))]
        npass = sum(o["pass"] for o in outs)
        res["runs"] += outs
        # A `fail` strategy normally has to fail on EVERY salt.  Strategies listed in NOISY_FAIL are the
        # exception: they guess a discrete answer at random, so they are accidentally right on a
        # predictable fraction of salts and a single pass is the guess being lucky, not a leak.  They are
        # held to a RATE instead - at most half the salts, which for a 1-in-3 guess is a wide but finite
        # allowance and still catches a strategy that passes systematically.  The alternative, letting
        # them hardcode one answer to stay deterministic, is what made `fix_all` accidentally correct on
        # every capacity-cause instance and get reported as a leak in seeds 25/110/208.
        if expect == "pass":
            good = npass >= max(1, int(np.ceil(0.75 * len(outs))))
        elif name in getattr(mod, "NOISY_FAIL", ()):
            good = npass <= len(outs) // 2
        else:
            good = npass == 0
        if not good:
            bad.append(name)
        if verbose:
            det = "; ".join("%s s=%s %s%s" % ("P" if o["pass"] else "F", o["score"],
                                               ",".join(k for k, v in o["items"].items() if not v["ok"]),
                                               (" ERR " + o["error"]) if o["error"] else "") for o in outs)
            print("  %-12s expect=%s pass=%d/%d %s | %s" % (name, expect, npass, len(outs), "OK " if good else "BAD", det))
    res["bad"] = bad
    res["accept"] = not bad
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    mod = importlib.import_module("l15.tasks." + a[0])
    seed = int(a[1])
    ns = int(a[a.index("--salts") + 1]) if "--salts" in a else 4
    r = gate(mod, seed, ns)
    if "--json" in a:
        json.dump(r, open(a[a.index("--json") + 1], "w"), indent=1, default=float)
    print("ACCEPT" if r["accept"] else "REJECT", r.get("bad"))
