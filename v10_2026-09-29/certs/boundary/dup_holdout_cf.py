"""Boundary check for a scaleup_l2 failure whose repo tunes wd on a random in-job holdout.
  python3 dup_holdout_cf.py <run_dir> <seed_instance> [replica_seed] [feature_matrix_name, default X]
Runs (a) the shipped repo and (b) the same repo with ONLY the holdout split made duplicate-aware (whole groups of
identical rows go to validation), both through the grader's run_job (same sandbox, data, CPU cap)."""
import os, sys, shutil, tempfile, json
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # v10 root
from l15.tasks import scaleup_l2 as L2, scaleup as B
rd, inst = sys.argv[1], int(sys.argv[2]); rs = int(sys.argv[3]) if len(sys.argv) > 3 else 0
FEAT = sys.argv[4] if len(sys.argv) > 4 else "X"      # name of the feature matrix at the split line
p = L2.sample_params(inst)
orig = os.path.join(rd, "app", "repo")
cf = tempfile.mkdtemp(prefix="cf_"); shutil.rmtree(cf); shutil.copytree(orig, cf, ignore=shutil.ignore_patterns("*.npy", "*.npz", "__pycache__"))
src = open(os.path.join(cf, "train.py")).read()
import re
# replace ONLY the assignment that splits perm into (validation, train); keep names, indentation, n_val
m = re.search(r"^(\s*)(\w+), (\w+) = perm\[:(\w+)\], perm\[\4:\]\s*$", src, re.M)
assert m and len(re.findall(r"= perm\[:(\w+)\], perm\[\1:\]", src)) == 1, "split line not found"
ind, va_n, tr_n, nv = m.group(1), m.group(2), m.group(3), m.group(4)
new = (ind + "_u, _inv = np.unique(%s, axis=0, return_inverse=True); _inv = np.asarray(_inv).ravel()\n" % FEAT +
       ind + "_gp = rng.permutation(len(_u)); _m = int(round(%s * len(_u) / n))\n" % nv +
       ind + "_hold = np.isin(_inv, _gp[:_m]); %s, %s = np.where(_hold)[0], np.where(~_hold)[0]" % (va_n, tr_n))
src_new = src[:m.start()] + new + src[m.end():]
open(os.path.join(cf, "train.py"), "w").write(src_new)
work = tempfile.mkdtemp(prefix="bnd_"); B._write_data(work, p)
with ThreadPoolExecutor(2) as ex:
    res = list(ex.map(lambda r: B.run_job(r, p, rs, data_dir=work), (orig, cf)))
for tag, (acc, info) in zip(("shipped", "dedup_holdout"), res):
    sel = [l for l in info.get("log_tail", "").splitlines() if any(w in l for w in ("candidate", "select", "chosen", "ensembl"))]
    print(json.dumps({"run": os.path.basename(rd.rstrip("/")), "instance": inst, "replica_seed": rs, "variant": tag, "acc": acc, "cpu_s": info.get("cpu_s"), "T": p["T"], "err": info.get("error"), "log": sel}))
shutil.rmtree(work, ignore_errors=True); shutil.rmtree(cf, ignore_errors=True)
