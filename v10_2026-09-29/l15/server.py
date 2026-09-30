"""L1.5 lab service (host side).  One HTTP endpoint, POST /api {"token","op","args"}.
The token (written into the run's /app/.lab_token by launch.py) maps to RUN_ROOT/<run>/lab.json, which names
the task directory; the hidden world is read from <task_dir>/hidden/world.json, which is never inside a sandbox.
Started in-process by harness/drive.py (dies with it); sessions are rebuilt from lab_ledger.jsonl."""
import glob, http.server, json, os, socketserver, threading, traceback
from .core import LabError, Session, load_world

PORT = int(os.environ.get("L15_PORT", "18787"))
RUN_ROOT = os.environ.get("LAB_RUN_ROOT", "/tmp/v10/runs")
_lock = threading.Lock()
_sessions = {}      # token -> (Session, lock)
_tokens = {}        # token -> run_dir


def _scan():
    for lj in glob.glob(os.path.join(RUN_ROOT, "*", "lab.json")):
        try:
            d = json.load(open(lj))
        except Exception:
            continue
        _tokens[d["token"]] = os.path.dirname(lj)


def session_for(token):
    with _lock:
        if token not in _tokens:
            _scan()
        rd = _tokens.get(token)
        if rd is None:
            raise LabError("unknown lab token")
        if token not in _sessions:
            meta = json.load(open(os.path.join(rd, "lab.json")))
            w = load_world(meta["task_dir"])
            _sessions[token] = (Session(w, os.path.join(rd, "lab_ledger.jsonl"), os.path.join(rd, "app")), threading.Lock())
        return _sessions[token]


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            sess, lk = session_for(req.get("token", ""))
            with lk:
                out = sess.call(req.get("op", ""), req.get("args") or {})
            self._send(200, {"ok": True, "result": out})
        except LabError as e:
            self._send(200, {"ok": False, "error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"ok": False, "error": "internal lab error (%s); the call was not charged" % type(e).__name__})


class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start():
    """Bind the lab port; if another driver already holds it, keep retrying in the background so that this
    process takes over the moment the other one exits (parallel drivers sharing one RUN_ROOT)."""
    import time as _t
    def _bind():
        while True:
            try:
                srv = Srv(("127.0.0.1", PORT), H)
            except OSError:
                _t.sleep(0.3); continue
            srv.serve_forever(); return
    threading.Thread(target=_bind, daemon=True).start()
