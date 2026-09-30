"""Local self-check for family O: the verifier must reward the oracle and reject every impostor.

Each probe gets its own copy of the task directory, because the verifier deletes the answer key from
disk before running agent code.
"""
import json, os, shutil, subprocess, sys, tempfile

TASK = sys.argv[1]
FAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fam_o")
PROBES = sys.argv[2:] or ["oracle", "spec_copy", "bestfit", "cap_oldest", "missing", "identity", "read_key"]


def probe_source(name):
    if name == "oracle":
        return open(os.path.join(FAM, "fast_pack.py")).read()
    if name == "spec_copy":                              # literally the specification: correct but slow
        return open(os.path.join(TASK, "environment/app/pack_ref.py")).read()
    if name == "missing":
        return None
    if name == "identity":                               # trivial nonsense
        return "def pack(lengths, params):\n    return list(range(len(lengths)))\n"
    if name == "read_key":                               # tries to read the answer key from /tests
        return ("import numpy as np, os, json\n"
                "def pack(lengths, params):\n"
                "    z = np.load('/tests/expected.npz')\n"
                "    for k in z.files:\n"
                "        if len(z[k]) == len(lengths): return [int(v) for v in z[k]]\n"
                "    return [0]*len(lengths)\n")
    # otherwise: a plausible-but-wrong fast solver = oracle with one semantic detail changed
    src = open(os.path.join(FAM, "fast_pack.py")).read()
    if name == "bestfit":
        return src.replace("b = leftmost_ge(L)", "b = best_fit(L)").replace(
            "    out = []",
            "    def best_fit(L):\n"
            "        c = [i for i in range(nbins) if isopen[i] and rem[i] >= L]\n"
            "        return min(c, key=lambda i: (rem[i], i)) if c else -1\n\n    out = []")
    if name == "cap_oldest":
        return src.replace("            v = argmin_open()", "            v = oldest\n"
                           "            while not isopen[v]: v += 1")
    raise SystemExit("unknown probe " + name)


out = {}
for name in PROBES:
    work = tempfile.mkdtemp()
    d = os.path.join(work, "t")
    shutil.copytree(TASK, d)
    app = os.path.join(d, "environment/app")
    src = probe_source(name)
    if src is None:
        pass
    else:
        open(os.path.join(app, "solution.py"), "w").write(src)
    env = dict(os.environ, TESTS=os.path.join(d, "tests"), APP=app, REWARD_DIR=os.path.join(work, "logs"))
    p = subprocess.run([sys.executable, os.path.join(d, "tests/verify_o.py")], env=env,
                       capture_output=True, text=True, timeout=900)
    try:
        r = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        r = {"reward": None, "stdout": p.stdout[-300:], "stderr": p.stderr[-300:]}
    det = r.get("detail", {}) or {}
    out[name] = {"reward": r.get("reward"), "big_s": det.get("big_seconds"),
                 "why": (det.get("failed") or [{}])[0].get("why"),
                 "case": (det.get("failed") or [{}])[0].get("case"),
                 "limit": det.get("time_limit_s")}
    shutil.rmtree(work, ignore_errors=True)
    print(name, json.dumps(out[name]), flush=True)

want = {"oracle": 1}
bad = [k for k, v in out.items() if v["reward"] != want.get(k, 0)]
print(json.dumps({"selfcheck_pass": not bad, "unexpected": bad}))
