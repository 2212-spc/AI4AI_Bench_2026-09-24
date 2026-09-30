"""End-to-end self-check for family V: run the shipped solution in a real /app, then probe impostors."""
import json, os, shutil, subprocess, sys, tempfile
TASK = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandbox import bwrap                                                         # noqa: E402


def grade(app, tests):
    p, _ = bwrap(app, tests, ["python3", "/tests/verify_v.py"])
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"reward": None, "stderr": p.stderr[-500:], "stdout": p.stdout[-300:]}


WORK = tempfile.mkdtemp()
APP0 = os.path.join(WORK, "app0")
TESTS = os.path.join(WORK, "tests")
shutil.copytree(os.path.join(TASK, "environment/app"), APP0)
shutil.copytree(os.path.join(TASK, "tests"), TESTS)
for f in os.listdir(os.path.join(TASK, "solution")):
    s = os.path.join(TASK, "solution", f)
    if os.path.isfile(s):
        shutil.copy(s, os.path.join(APP0, f))
r, _ = bwrap(APP0, None, ["bash", "/app/solve.sh"])
if r.returncode:
    print("solve.sh FAILED", (r.stderr or "")[-800:]); sys.exit(1)
oracle = grade(APP0, TESTS)
print("oracle", json.dumps({k: oracle.get(k) for k in ("reward",)}), oracle["detail"]["n_correct"], "/",
      oracle["detail"]["n_claims"])
REF = json.load(open(os.path.join(APP0, "findings.json")))
KEY = json.load(open(os.path.join(TASK, "tests/key.json")))
LONG = "the module cannot violate this claim because the accumulator is checked before the append " \
       "and the oversize case is clamped, so every emitted batch is bounded by construction."


def fresh(findings=None, mutate=None):
    d = os.path.join(WORK, "p%d" % fresh.n); fresh.n += 1
    shutil.copytree(APP0, d)
    if findings is not None:
        json.dump(findings, open(os.path.join(d, "findings.json"), "w"))
    if mutate:
        mutate(d)
    return d


fresh.n = 0
probes = {}

probes["all_violated_garbage"] = {c: {"status": "violated", "witness": "w.json"} for c in KEY}
probes["all_holds"] = {c: {"status": "holds", "argument": LONG} for c in KEY}
probes["right_labels_no_witness"] = {c: ({"status": "violated", "witness": "nope.json"}
                                         if KEY[c] == "violated" else {"status": "holds", "argument": LONG})
                                     for c in KEY}
five = dict(REF); five["V1"] = {"status": "holds", "argument": LONG}
probes["five_of_six_V1_conceded"] = five
five2 = dict(REF); five2["V4"] = {"status": "violated", "witness": "witnesses/v3.json"}
probes["five_of_six_V4_attacked"] = five2
esc = dict(REF); esc["V3"] = {"status": "violated", "witness": "../tests/key.json"}
probes["witness_path_escape"] = esc
sh = dict(REF); sh["V4"] = {"status": "holds", "argument": "it is fine"}
probes["short_argument"] = sh
probes["missing_status"] = {c: {"witness": "witnesses/v1.json"} for c in KEY}

for name, f in probes.items():
    g = grade(fresh(f), TESTS)
    print("%-26s reward=%s correct=%s" % (name, g.get("reward"),
                                          (g.get("detail") or {}).get("n_correct")))

# no findings.json at all
d = fresh(); os.remove(os.path.join(d, "findings.json"))
print("%-26s reward=%s" % ("missing_findings", grade(d, TESTS).get("reward")))

# the agent rewrites the module so that its own copy violates V4, then claims V4
def tamper(d):
    p = os.path.join(d, "dataprep/pack.py")
    open(p, "w").write("def pack(lengths, max_tokens):\n    return [list(range(len(lengths)))]\n")
    json.dump(dict(REF, V4={"status": "violated", "witness": "witnesses/v4.json"}),
              open(os.path.join(d, "findings.json"), "w"))
    json.dump({"lengths": [1, 1, 1], "max_tokens": 2}, open(os.path.join(d, "witnesses/v4.json"), "w"))
print("%-26s reward=%s" % ("tampered_module", grade(fresh(mutate=tamper), TESTS).get("reward")))

# an oversized witness file
def big(d):
    json.dump(dict(REF, V3={"status": "violated", "witness": "witnesses/big.json"}),
              open(os.path.join(d, "findings.json"), "w"))
    open(os.path.join(d, "witnesses/big.json"), "w").write("[" + "0," * 1_200_000 + "0]")
print("%-26s reward=%s" % ("oversized_witness", grade(fresh(mutate=big), TESTS).get("reward")))

# a plausible, in-scope but unconfirmed V1 witness: a genuinely similar pair that dedup does catch
def nearp(d):
    import random
    sys.path.insert(0, os.path.join(TASK, "solution"))
    from make_witnesses import WORDS
    rng = random.Random(5)
    base = [rng.choice(WORDS) for _ in range(600)]
    b = list(base)
    for i in rng.sample(range(600), 8):
        b[i] = "z%d" % i
    json.dump({"docs": [" ".join(base), " ".join(b)]}, open(os.path.join(d, "witnesses/near.json"), "w"))
    json.dump(dict(REF, V1={"status": "violated", "witness": "witnesses/near.json"}),
              open(os.path.join(d, "findings.json"), "w"))
g = grade(fresh(mutate=nearp), TESTS)
print("%-26s reward=%s why=%s" % ("plausible_V1_pair", g.get("reward"),
                                  (g.get("detail") or {})["per_claim"]["V1"]["why"][:60]))

# grading asymmetry: a confirmed witness must be accepted even when the key says the claim holds
T2 = os.path.join(WORK, "tests_flipped")
shutil.copytree(TESTS, T2)
json.dump(dict(KEY, V3="holds"), open(os.path.join(T2, "key.json"), "w"))
g = grade(fresh(REF), T2)
print("%-26s reward=%s  (a witness the authors did not expect is still credited)"
      % ("key_says_holds_but_proved", g.get("reward")))

print("SELFCHECK_V oracle_reward=%s" % oracle.get("reward"))
