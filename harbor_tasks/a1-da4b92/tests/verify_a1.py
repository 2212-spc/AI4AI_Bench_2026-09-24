import json, os, sys
sys.path.insert(0, "/tests")
import a1_verify_core as V
d = json.load(open(os.environ.get("SPEC", "/tests/verify_spec.json")))
ok, rep = V.evaluate(os.environ.get("APP", "/app"), d["spec"], V.teacher_from(d["teacher"]), verbose=False)
rd = os.environ.get("REWARD_DIR", "/logs/verifier"); os.makedirs(rd, exist_ok=True)
open(os.path.join(rd, "reward.txt"), "w").write("%d\n" % int(ok)); print(json.dumps(rep)[:2000])
