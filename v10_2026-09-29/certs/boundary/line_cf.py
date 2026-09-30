"""Single-edit counterfactual for a scaleup_l2 failure.
  python3 line_cf.py <run_dir> <seed_instance> <tag> <file> <old> <new>
<old> must occur exactly once in <file> of the shipped repo; it is replaced by <new> (nothing else changes).
The edited repo is run for replica seeds 0 and 1 (the production pair) through the grader's run_job
(same sandbox, data, CPU cap); prints one JSON line per replica plus the mean."""
import os, sys, shutil, tempfile, json
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # v10 root
from l15.tasks import scaleup_l2 as L2, scaleup as B
rd, inst, tag, fn, old, new = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
p = L2.sample_params(inst)
orig = os.path.join(rd, "app", "repo")
cf = tempfile.mkdtemp(prefix="cf_"); shutil.rmtree(cf); shutil.copytree(orig, cf, ignore=shutil.ignore_patterns("*.npy", "*.npz", "__pycache__"))
src = open(os.path.join(cf, fn)).read()
assert src.count(old) == 1, "edit target must occur exactly once (found %d)" % src.count(old)
open(os.path.join(cf, fn), "w").write(src.replace(old, new))
work = tempfile.mkdtemp(prefix="bnd_"); B._write_data(work, p)
with ThreadPoolExecutor(2) as ex:
    res = list(ex.map(lambda r: B.run_job(cf, p, r, data_dir=work), (0, 1)))
accs = []
for rs, (acc, info) in zip((0, 1), res):
    accs.append(acc)
    sel = [l for l in info.get("log_tail", "").splitlines() if any(w in l for w in ("candidate", "select", "chosen", "ensembl", "member", "weight_decay"))]
    print(json.dumps({"run": os.path.basename(rd.rstrip("/")), "instance": inst, "replica_seed": rs, "variant": tag,
                      "edit": {"file": fn, "old": old, "new": new}, "acc": acc, "cpu_s": info.get("cpu_s"), "T": p["T"],
                      "err": info.get("error"), "log": sel}))
ok = all(a is not None for a in accs)
print(json.dumps({"run": os.path.basename(rd.rstrip("/")), "variant": tag, "mean_acc": (sum(accs) / 2 if ok else None), "T": p["T"],
                  "pass": (ok and sum(accs) / 2 >= p["T"])}))
shutil.rmtree(work, ignore_errors=True); shutil.rmtree(cf, ignore_errors=True)
