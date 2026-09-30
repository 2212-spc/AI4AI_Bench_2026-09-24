"""Single-tenant budgeted lab sidecar for Harbor E1 tasks.

Runs in its own container (service `lab` in environment/docker-compose.yaml). The hidden world definition
(world.json) lives only in this image; the agent container sees nothing but the `lab` CLI and the HTTP answers.
Budget is enforced here (HTTP 429 after LAB_BUDGET runs). The ledger stays inside this container.

env: LAB_WORLD (default /srv/lab/world.json), LAB_BUDGET (default 40), LAB_LEDGER (default /var/lab/ledger.jsonl),
     LAB_PORT (default 18777), LAB_BIND (default 0.0.0.0)"""
import json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import e1_world as W

WORLD_JSON = os.environ.get("LAB_WORLD", "/srv/lab/world.json")
BUDGET = int(os.environ.get("LAB_BUDGET", "40"))
LEDGER = os.environ.get("LAB_LEDGER", "/var/lab/ledger.jsonl")
PORT = int(os.environ.get("LAB_PORT", "18777"))
BIND = os.environ.get("LAB_BIND", "0.0.0.0")
_lock = threading.Lock()
_world = None


def world():
    global _world
    if _world is None:
        _world = W.build_world(json.load(open(WORLD_JSON)))
    return _world


def ledger():
    if not os.path.exists(LEDGER): return []
    return [json.loads(l) for l in open(LEDGER) if l.strip()]


def handle(req):
    op = req.get("op")
    with _lock:
        led = ledger(); used = len(led)
        if op == "status": return 200, {"budget": BUDGET, "used": used, "remaining": BUDGET - used}
        if op == "history": return 200, {"runs": [{k: v for k, v in r.items() if k != "t"} for r in led]}
        if op != "run": return 400, {"error": "op must be run|status|history"}
        changes = req.get("changes") or []
        if isinstance(changes, str): changes = [c for c in changes.split(",") if c]
        bad = [c for c in changes if c not in W.NAMES]
        if bad: return 400, {"error": "unknown change(s) %s; valid: %s" % (bad, W.NAMES)}
        seed = req.get("seed")
        if seed is None: seed = int(np.random.default_rng().integers(0, 1_000_000))
        try: seed = int(seed)
        except Exception: return 400, {"error": "seed must be an integer"}
        if not 0 <= seed < 1_000_000: return 400, {"error": "seed must be in [0, 1000000)"}
        if used >= BUDGET: return 429, {"error": "lab budget exhausted (%d/%d runs used)" % (used, BUDGET)}
        changes = sorted(set(changes), key=W.NAMES.index)
        v, div = W.run_cell(world(), W.mask_of(changes), seed)
        rec = {"run": used + 1, "changes": changes, "seed": seed, "val_mse": None if div else round(v, 6),
               "diverged": bool(div), "t": time.time()}
        os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
        with open(LEDGER, "a") as f: f.write(json.dumps(rec) + "\n")
        out = dict(rec); out.pop("t"); out["remaining"] = BUDGET - used - 1
        return 200, out


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0)); code, body = handle(json.loads(self.rfile.read(n) or b"{}"))
        except Exception as e:
            code, body = 500, {"error": repr(e)}
        b = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def log_message(self, *a):
        pass


def serve(port=PORT, bind=BIND, background=False):
    world()  # build data once before accepting requests
    srv = ThreadingHTTPServer((bind, port), H); srv.daemon_threads = True
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start(); return srv
    srv.serve_forever()


if __name__ == "__main__":
    serve()
