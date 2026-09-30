"""ScaleLab lab service (host side).  The agent reaches it only through bin/lab with a per-run token.

Registry: every <RUN_ROOT>/<run>/lab.json {"token": ..., "task_dir": ...} is served.
State:    <run>/lab_ledger.jsonl (server side, outside the sandbox) holds every accepted request; on
          (re)start the Session is rebuilt by replaying the ledger through Session.run, which is
          deterministic, and each replayed result is checked against the logged one.
Execution goes through scalelab.lab.Session - the code the builder's oracle, rivals and gates used."""
import glob, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scalelab.lab import Session, LabError

PORT = 18777
RUN_ROOT = os.environ.get("LAB_RUN_ROOT", "/tmp/v3/runs")
_lock = threading.Lock()
_sessions = {}


def _registry():
    reg = {}
    for p in glob.glob(RUN_ROOT + "/*/lab.json"):
        try:
            d = json.load(open(p)); d["run_dir"] = os.path.dirname(p); reg[d["token"]] = d
        except Exception:
            pass
    return reg


def _public_spec(spec):
    return {k: spec[k] for k in ("knobs", "fixed", "caps", "metrics") if k in spec}


def _session(ent):
    tok = ent["token"]
    if tok in _sessions:
        return _sessions[tok]
    w = json.load(open(os.path.join(ent["task_dir"], "hidden", "world.json")))
    s = Session(w["params"], w["spec"], w["salt"])
    led = os.path.join(ent["run_dir"], "lab_ledger.jsonl")
    if os.path.exists(led):
        for line in open(led):
            if not line.strip():
                continue
            rec = json.loads(line)
            res = s.run(rec["req"])
            if json.dumps(res, sort_keys=True) != json.dumps(rec["result"], sort_keys=True):
                raise RuntimeError("ledger replay mismatch in %s" % ent["run_dir"])
    _sessions[tok] = s
    return s


def _run_one(ent, s, req):
    if not isinstance(req, dict):
        raise LabError("a run request must be a JSON object of knob values")
    res = s.run(req)
    with open(os.path.join(ent["run_dir"], "lab_ledger.jsonl"), "a") as f:
        f.write(json.dumps({"req": req, "result": res, "t": time.time()}) + "\n")
    return res


def handle(req):
    ent = _registry().get(req.get("token"))
    if ent is None:
        return 403, {"error": "unknown token"}
    op = req.get("op")
    with _lock:
        s = _session(ent)
        if op == "spec":
            return 200, _public_spec(s.spec)
        if op == "status":
            return 200, s.status()
        if op == "history":
            return 200, {"runs": s.log}
        if op == "run":
            try:
                return 200, _run_one(ent, s, req.get("req") or {})
            except LabError as e:
                return 400, {"error": str(e), "status": s.status()}
        if op == "batch":
            out = []
            reqs = req.get("reqs") or []
            if not isinstance(reqs, list) or len(reqs) > 200:
                return 400, {"error": "reqs must be a list of at most 200 requests"}
            for i, r in enumerate(reqs):
                try:
                    out.append(_run_one(ent, s, r))
                except LabError as e:
                    return 200, {"results": out, "error": "request %d refused: %s" % (i, e), "status": s.status()}
            return 200, {"results": out, "error": None, "status": s.status()}
        return 400, {"error": "op must be spec|run|batch|status|history"}


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
            code, body = handle(req)
        except Exception as e:
            code, body = 500, {"error": "lab service error: %r" % (e,)}
        b = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def log_message(self, *a):
        pass


def start(port=PORT):
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    start(); print("lab server on", PORT, "root", RUN_ROOT); sys.stdout.flush()
    time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 1e9)
