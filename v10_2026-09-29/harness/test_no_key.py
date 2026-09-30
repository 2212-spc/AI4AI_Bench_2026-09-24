"""Security test for the v7 harness: intercept every launch path and assert the real key never reaches a sandbox.

Uses a FAKE secret dir (never the operator's), a fake upstream that records the headers it receives, and
mocks for subprocess/urllib, so it runs offline.  Checks:
  1. gateway swaps the placeholder for the (fake) real key on the way out, and only there;
  2. cc_agent.tick: bwrap argv (incl. --setenv values) and the Popen env contain no key;
  3. gpt_agent.tick: the API request carries only the placeholder; tool-command bwrap argv contains no key;
  4. l15 runner: submitted code runs with --unshare-all and --clearenv, and argv contains no key;
  5. launch.py writes no key into the run dir.
Exit code 0 = all pass."""
import http.server, json, os, sys, tempfile, threading, subprocess, io
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
FAKE = "sk-FAKE-REAL-KEY-0123456789abcdefghijKLMN"
fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


# fake upstream
seen = []


class Up(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); self.rfile.read(n)
        seen.append(dict(self.headers))
        b = json.dumps({"output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}], "usage": {}}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    do_GET = do_POST


up = http.server.HTTPServer(("127.0.0.1", 0), Up)
threading.Thread(target=up.serve_forever, daemon=True).start()
sec = tempfile.mkdtemp()
open(os.path.join(sec, "key"), "w").write(FAKE)
open(os.path.join(sec, "url"), "w").write("http://127.0.0.1:%d" % up.server_address[1])
os.environ["SEC_DIR"] = sec
os.environ.pop("GATEWAY_URL", None)
import importlib, gateway
importlib.reload(gateway)
url = gateway.start()
check(FAKE not in json.dumps(dict(os.environ)), "driver environment holds no key after gateway.start()")
import urllib.request
r = urllib.request.urlopen(urllib.request.Request(url + "/v1/responses", data=b"{}", headers={"Authorization": "Bearer " + gateway.PLACEHOLDER}))
r.read()
check(seen and seen[-1].get("Authorization") == "Bearer " + FAKE, "gateway injects the real key upstream")

# 5. launch
rr = tempfile.mkdtemp(); os.environ["LAB_RUN_ROOT"] = rr
task = tempfile.mkdtemp(); os.makedirs(task + "/environment/app/bin"); open(task + "/instruction.md", "w").write("do nothing")
import launch
importlib.reload(launch)
rd_cc = launch.launch(task, "cc", "claude-x", 0, "t_cc")
rd_gpt = launch.launch(task, "gpt", "gpt-x", 0, "t_gpt")
blob = ""
for root, _, files in os.walk(rr):
    for f in files:
        blob += open(os.path.join(root, f), errors="replace").read()
check(FAKE not in blob, "launch writes no key into run dirs")

# 2. cc_agent tick with mocked Popen
import cc_agent, time
cap = {}


class FakeP:
    def __init__(self, argv, **kw):
        cap["argv"] = argv; cap["env"] = kw.get("env"); self.pid = os.getpid()
        kw["stdout"].write(json.dumps({"type": "result", "subtype": "success", "result": "ok", "total_cost_usd": 0}).encode() + b"\n")

    def wait(self, timeout=None): return 0


orig = subprocess.Popen
cc_agent.subprocess.Popen = FakeP
cc_agent.tick(rd_cc, time.time() + 100)
cc_agent.subprocess.Popen = orig
check("argv" in cap and FAKE not in json.dumps(cap["argv"]), "cc_agent bwrap argv has no key")
check(cap.get("env") is None or FAKE not in json.dumps(cap["env"]), "cc_agent Popen env has no key")
check("--clearenv" in cap.get("argv", []), "cc_agent sandbox uses --clearenv")
i = cap["argv"].index("ANTHROPIC_API_KEY") if "ANTHROPIC_API_KEY" in cap.get("argv", []) else -1
check(i > 0 and cap["argv"][i + 1] == gateway.PLACEHOLDER, "cc_agent passes only the placeholder as ANTHROPIC_API_KEY")

# 3. gpt_agent tick: API request headers + tool argv
import gpt_agent
importlib.reload(gpt_agent)
seen.clear()
st = json.load(open(rd_gpt + "/state.json"))
st["items"].append({"type": "function_call", "call_id": "c1", "name": "shell", "arguments": json.dumps({"command": ["bash", "-lc", "env"]})})
json.dump(st, open(rd_gpt + "/state.json", "w"))
capr = {}
orig_run = subprocess.run


def fake_run(argv, **kw):
    capr["argv"] = argv
    return orig_run(["true"], capture_output=True)


gpt_agent.subprocess.run = fake_run
gpt_agent.tick(rd_gpt, time.time() + 100)
gpt_agent.subprocess.run = orig_run
check("argv" in capr and FAKE not in json.dumps(capr["argv"]), "gpt_agent tool bwrap argv has no key")
check(seen and seen[-1].get("Authorization") == "Bearer " + FAKE, "gpt_agent reached upstream only via gateway swap")
tr = open(rd_gpt + "/traj.jsonl").read() + open(rd_gpt + "/state.json").read()
check(FAKE not in tr, "gpt_agent state/trajectory hold no key")

# 4. runner argv
from l15 import runner
capn = {}


class FakeP2:
    def __init__(self, argv, **kw):
        capn["argv"] = argv; capn["env"] = kw.get("env"); self.pid = 999999999
        self.stdout = io.BytesIO(b""); self.stdin = io.BytesIO(); self.returncode = 0

    def poll(self): return 0

    def wait(self, timeout=None): return 0


class D:
    finished = False
    def handle(self, m): return {}


runner.subprocess.Popen = FakeP2
runner.select.select = lambda r, w, x, t: (r, [], [])
sub = tempfile.mkdtemp(); open(sub + "/policy.py", "w").write("def run(env): pass\n")
lib = tempfile.mktemp(suffix=".py"); open(lib, "w").write("")
try:
    runner.run(sub, "policy", "run", lib, D(), 2)
except Exception as e:
    print("runner mock raised", type(e).__name__, e)
runner.subprocess.Popen = orig
a = capn.get("argv", [])
check(a and "--unshare-all" in a and "--clearenv" in a and FAKE not in json.dumps(a), "runner sandbox: no net, clean env, no key")
print("RESULT:", "ALL PASS" if not fails else "%d FAIL" % len(fails))
sys.exit(1 if fails else 0)
