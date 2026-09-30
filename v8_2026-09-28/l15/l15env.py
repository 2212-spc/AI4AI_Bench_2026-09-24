"""Generic client library for submitted algorithms (copied into the runner sandbox as /lib/l15env.py).

Protocol: JSON lines.  The submission's stdout is redirected to stderr so stray prints cannot corrupt it."""
import json, sys

_out = sys.stdout
sys.stdout = sys.stderr
_in = sys.stdin


def _send(msg):
    _out.write(json.dumps(msg) + "\n"); _out.flush()


def _call(op, **a):
    _send(dict(op=op, **a))
    line = _in.readline()
    if not line:
        raise SystemExit("host closed the connection")
    r = json.loads(line)
    if isinstance(r, dict) and r.get("error"):
        raise RuntimeError(r["error"])
    return r


def _finish():
    _send({"op": "__done__"})


class Env:
    """Handle passed to your entry function.  Task-specific methods are documented in the task's docs."""

    def __init__(self):
        self.info = _call("init")
        for k, v in self.info.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def call(self, op, **args):
        return _call(op, **args)
