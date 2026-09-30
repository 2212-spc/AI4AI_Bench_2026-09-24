"""Grade a finished agent run directory against a task's tests, in the same sandbox shape harbor uses."""
import json, os, shutil, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandbox import bwrap                                                         # noqa: E402

VERIFY = {"o-pack-a": "verify_o.py", "c-audit-a": "verify_c.py", "v-falsify-a": "verify_v.py"}
run, task = sys.argv[1], sys.argv[2]
slug = os.path.basename(task.rstrip("/"))
W = tempfile.mkdtemp()
APP, TESTS = os.path.join(W, "app"), os.path.join(W, "tests")
shutil.copytree(os.path.join(run, "app"), APP)
shutil.copytree(os.path.join(task, "tests"), TESTS)
p, _ = bwrap(APP, TESTS, ["bash", "/tests/test.sh"], timeout=1200)
line = [l for l in p.stdout.splitlines() if l.strip().startswith("{")]
out = json.loads(line[-1]) if line else {"reward": None, "stderr": p.stderr[-500:]}
print(json.dumps({"run": os.path.basename(run.rstrip("/")), "task": slug, "reward": out.get("reward"),
                  "detail": out.get("detail")}, ensure_ascii=False)[:4000])
