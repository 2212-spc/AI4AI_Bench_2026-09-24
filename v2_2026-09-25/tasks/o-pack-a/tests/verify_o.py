"""Family O verifier (separate container, no network). Reads only the agent artifact /app/solution.py.

Truth = tests/expected.npz, produced at authoring time by executing the specification implementation.
The hidden inputs and expected outputs are loaded into memory and then deleted from disk before any
agent-authored code is executed, so a submission cannot read its own answer key.
"""
import json, os, shutil, subprocess, sys, tempfile, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_inputs import make_lengths                                   # noqa: E402

TESTS = os.environ.get("TESTS", "/tests")
APP = os.environ.get("APP", "/app")
REWARD_DIR = os.environ.get("REWARD_DIR", "/logs/verifier")

RUNNER = r'''
import json, sys, time
import numpy as np
import solution
inp = np.load(sys.argv[1]); params = json.load(open(sys.argv[2]))
t = time.time(); out = solution.pack([int(v) for v in inp], params); el = time.time() - t
np.save(sys.argv[3], np.asarray(out, dtype=np.int64))
open(sys.argv[4], "w").write("%.4f" % el)
'''


def calibrate():
    t = time.process_time()
    x = 0
    for i in range(20_000_000):
        x += i % 7
    return (time.process_time() - t)


def finish(ok, detail):
    os.makedirs(REWARD_DIR, exist_ok=True)
    open(os.path.join(REWARD_DIR, "reward.txt"), "w").write("1" if ok else "0")
    print(json.dumps({"reward": 1 if ok else 0, "detail": detail}, default=str)[:20000])
    sys.exit(0)


def main():
    spec = json.load(open(os.path.join(TESTS, "spec.json")))
    z = np.load(os.path.join(TESTS, "expected.npz"))
    expected = {k: z[k] for k in z.files}
    params = spec["params"]

    def inputs_of(c):
        return c["lengths"] if "lengths" in c else make_lengths(c["profile"], c["n"], params["capacity"], c["seed"])

    cases = [("c%03d" % i, inputs_of(c), c) for i, c in enumerate(spec["suite"])]
    cases.append(("big", inputs_of(spec["big"]), spec["big"]))

    work = tempfile.mkdtemp()
    sol = os.path.join(work, "sol")
    os.makedirs(sol)
    src = os.path.join(APP, "solution.py")
    if not os.path.exists(src):
        finish(False, {"error": "no /app/solution.py"})
    text = open(src, errors="replace").read()
    shutil.copy(src, os.path.join(sol, "solution.py"))
    open(os.path.join(sol, "_runner.py"), "w").write(RUNNER)
    json.dump(params, open(os.path.join(work, "params.json"), "w"))

    # The agent's module is imported by a runner that lives on the same filesystem, so the key is taken
    # off disk before any agent code executes.  Both files are already fully in memory above.  Harbor
    # grades each submission in a fresh container, so removing them there costs nothing; when the task is
    # graded in place - re-scoring an archived run, or the self-check running eighteen probes against one
    # copy of the task - deleting the key would make the verifier single-use, which is how a benchmark
    # silently stops being reproducible.  HARBOR_CONTAINER is set by tests/test.sh inside the container.
    if os.environ.get("HARBOR_CONTAINER"):
        for p in ("expected.npz", "spec.json"):
            try:
                os.remove(os.path.join(TESTS, p))
            except OSError:
                pass

    calib = calibrate() / spec["calib_ref_s"]
    limit = spec["limit_s"] * min(3.0, max(0.7, calib))
    report = {"calibration": round(calib, 3), "time_limit_s": round(limit, 1), "failed": [], "n_cases": len(cases),
              "mentions_pack_ref": ("pack_ref" in text)}

    for name, L, meta in cases:
        fin = os.path.join(work, "in.npy"); fout = os.path.join(work, "out.npy"); ftim = os.path.join(work, "t.txt")
        np.save(fin, np.asarray(L, dtype=np.int64))
        for f in (fout, ftim):
            if os.path.exists(f):
                os.remove(f)
        to = max(60.0, 4 * limit) if name == "big" else 120.0
        try:
            p = subprocess.run([sys.executable, "_runner.py", fin, os.path.join(work, "params.json"), fout, ftim],
                               cwd=sol, capture_output=True, text=True, timeout=to)
        except subprocess.TimeoutExpired:
            report["failed"].append({"case": name, "why": "timeout>%.0fs" % to, "meta": meta})
            break
        if not os.path.exists(fout):
            report["failed"].append({"case": name, "why": "crash", "stderr": (p.stderr or "")[-400:], "meta": meta})
            break
        got = np.load(fout)
        exp = expected[name]
        if got.shape != exp.shape or not np.array_equal(got, exp):
            d = int(np.argmax(got[:len(exp)] != exp[:len(got)])) if len(got) == len(exp) else -1
            report["failed"].append({"case": name, "why": "output mismatch", "first_diff_index": d,
                                     "n": len(exp), "meta": {k: v for k, v in meta.items() if k != "lengths"}})
            break
        if name == "big":
            el = float(open(ftim).read())
            report["big_seconds"] = round(el, 2)
            if el > limit:
                report["failed"].append({"case": "big", "why": "too slow: %.1fs > %.1fs" % (el, limit)})

    finish(not report["failed"], report)


if __name__ == "__main__":
    main()
