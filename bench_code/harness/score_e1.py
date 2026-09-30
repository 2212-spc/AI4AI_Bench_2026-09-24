import json, sys, os
sys.path.insert(0, "/tmp/bench/gen"); sys.path.insert(0, "/tmp/bench/lab")
import e1_world as W
for rd in sys.argv[1:]:
    lab = json.load(open(rd + "/lab.json")); sc = json.load(open("/tmp/bench/instances/e1/%s/hidden/scoring.json" % lab["instance"]))
    mu = sc["mu"]; opt = sc["opt"]
    led = [json.loads(l) for l in open(rd + "/lab_ledger.jsonl")] if os.path.exists(rd + "/lab_ledger.jsonl") else []
    try:
        d = json.load(open(rd + "/app/decision.json")); m = W.mask_of(d["ship"]); pred = float(d["predicted_val_mse"])
    except Exception as e:
        print(os.path.basename(rd), "NO/INVALID decision", repr(e), "runs", len(led)); continue
    reg = mu[m] / mu[opt] - 1; ok_d = reg <= sc["delta_rel"]; perr = abs(pred - mu[m]) / mu[m]; ok_p = perr <= sc["pred_tol"]
    seeds = {}
    for r in led: seeds.setdefault(r["seed"], set()).add("+".join(r["changes"]) or "-")
    crn = sum(1 for s, c in seeds.items() if len(c) > 1)
    cells = len({"+".join(r["changes"]) for r in led})
    print("%-16s %s ship=%s regret=%.1f%% (δ=%.1f%%) pred_err=%.1f%% runs=%d distinct_cells=%d seeds_shared=%d" % (
        os.path.basename(rd), "PASS" if ok_d and ok_p else "FAIL", "+".join(d["ship"]) or "-", 100 * reg, 100 * sc["delta_rel"], 100 * perr, len(led), cells, crn))
