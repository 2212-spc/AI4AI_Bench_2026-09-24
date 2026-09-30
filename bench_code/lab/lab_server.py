"""Budgeted lab service for E-family tasks (host side; the agent only sees the `lab` CLI).

Runs as a daemon thread inside the tick driver (drive.py) so it is alive whenever agents are.
Registry: every run dir that contains lab.json {"token":..., "instance":..., "budget":...} is served.
Ledger: <run_dir>/lab_ledger.jsonl (append-only, server-side; never visible inside the sandbox).
In the Harbor export the same file is the entrypoint of a sidecar container."""
import glob, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/tmp/bench/gen"); sys.path.insert(0, "/tmp/bench/lab")
import numpy as np
import e1_world as W

PORT = 18777
INST_ROOT = "/tmp/bench/instances/e1"
RUN_ROOT = "/tmp/bench/runs"
_lock = threading.Lock()
_worlds = {}


def _registry():
    reg = {}
    for p in glob.glob(RUN_ROOT + "/*/lab.json"):
        try:
            d = json.load(open(p)); d["run_dir"] = os.path.dirname(p); reg[d["token"]] = d
        except Exception:
            pass
    return reg


def _world(inst):
    if inst not in _worlds:
        params = json.load(open(os.path.join(INST_ROOT, inst, "hidden", "world.json")))
        _worlds[inst] = W.build_world(params)
    return _worlds[inst]


def _ledger(run_dir):
    p = os.path.join(run_dir, "lab_ledger.jsonl")
    if not os.path.exists(p): return []
    return [json.loads(l) for l in open(p) if l.strip()]


def handle(req):
    reg = _registry()
    ent = reg.get(req.get("token"))
    if ent is None: return 403, {"error": "unknown token"}
    op = req.get("op")
    with _lock:
        led = _ledger(ent["run_dir"])
        used = len(led)
        if op == "status":
            return 200, {"budget": ent["budget"], "used": used, "remaining": ent["budget"] - used}
        if op == "history":
            return 200, {"runs": led}
        if op != "run": return 400, {"error": "op must be run|status|history"}
        changes = req.get("changes") or []
        if isinstance(changes, str): changes = [c for c in changes.split(",") if c]
        bad = [c for c in changes if c not in W.NAMES]
        if bad: return 400, {"error": "unknown change(s) %s; valid: %s" % (bad, W.NAMES)}
        seed = req.get("seed")
        if seed is None: seed = int(np.random.default_rng().integers(0, 1_000_000))
        seed = int(seed)
        if not 0 <= seed < 1_000_000: return 400, {"error": "seed must be in [0, 1000000)"}
        if used >= ent["budget"]:
            return 429, {"error": "lab budget exhausted (%d/%d runs used)" % (used, ent["budget"])}
        changes = sorted(set(changes), key=W.NAMES.index)
        v, div = W.run_cell(_world(ent["instance"]), W.mask_of(changes), seed)
        rec = {"run": used + 1, "changes": changes, "seed": seed,
               "val_mse": None if div else round(v, 6), "diverged": bool(div), "t": time.time()}
        with open(os.path.join(ent["run_dir"], "lab_ledger.jsonl"), "a") as f: f.write(json.dumps(rec) + "\n")
        out = dict(rec); out.pop("t"); out["remaining"] = ent["budget"] - used - 1
        return 200, out


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
            code, body = handle(req)
        except Exception as e:
            code, body = 500, {"error": repr(e)}
        b = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def log_message(self, *a):
        pass


def start(port=PORT):
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    return srv


if __name__ == "__main__":
    start(); print("lab server on", PORT); time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 1e9)
