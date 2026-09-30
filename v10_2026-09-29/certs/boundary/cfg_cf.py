"""Config-override counterfactual / feasibility probe for scaleup_l2 repos.
  python3 cfg_cf.py <run_dir|repo_dir> <seed_instance> <tag> '<json overrides>' [replicas=0,1]
Copies the shipped repo, updates config.json with the overrides (nothing else changes), and runs the
production replicas through the grader's run_job (same sandbox, data, CPU cap).  One JSON line per replica + mean."""
import os, sys, shutil, tempfile, json
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # v10 root
from l15.tasks import scaleup_l2 as L2, scaleup as B
rd, inst, tag, ov = sys.argv[1], int(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4])
reps = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "0,1").split(",")]
p = L2.sample_params(inst)
orig = os.path.join(rd, "app", "repo") if os.path.isdir(os.path.join(rd, "app", "repo")) else rd
cf = tempfile.mkdtemp(prefix="cf_"); shutil.rmtree(cf); shutil.copytree(orig, cf, ignore=shutil.ignore_patterns("*.npy", "*.npz", "__pycache__"))
cp = os.path.join(cf, "config.json"); c = json.load(open(cp)); c.update(ov); json.dump(c, open(cp, "w"), indent=2)
work = tempfile.mkdtemp(prefix="bnd_"); B._write_data(work, p)
with ThreadPoolExecutor(len(reps)) as ex:
    res = list(ex.map(lambda r: B.run_job(cf, p, r, data_dir=work), reps))
accs = []
for rs, (acc, info) in zip(reps, res):
    accs.append(acc)
    print(json.dumps({"run": os.path.basename(rd.rstrip("/")), "instance": inst, "replica_seed": rs, "variant": tag,
                      "overrides": ov, "acc": acc, "cpu_s": info.get("cpu_s"), "T": p["T"], "err": info.get("error")}))
ok = all(a is not None for a in accs)
print(json.dumps({"run": os.path.basename(rd.rstrip("/")), "variant": tag, "overrides": ov,
                  "mean_acc": (sum(accs) / len(accs) if ok else None), "T": p["T"], "pass": (ok and sum(accs) / len(accs) >= p["T"])}))
shutil.rmtree(work, ignore_errors=True); shutil.rmtree(cf, ignore_errors=True)
