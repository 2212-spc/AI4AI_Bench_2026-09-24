"""Client library for K6 schedulers (mounted read-only as /lib/l15env.py inside the scheduler sandbox).

    env.n_segments   number of segments in the run
    env.b_min        smallest batch you may request
    env.b_max        largest batch you may request
    env.t_overhead   fixed seconds of overhead per optimizer step
    env.throughput   examples/second of the accelerator (per-step time is t_overhead + batch/throughput)
    env.run_segment(batch) -> {"segment", "batch", "steps", "seconds", "segments_left", "done"}
    env.elapsed()    -> seconds of wall-clock used so far

You must call run_segment exactly n_segments times; the run ends when the last segment returns done=True.

Protocol: JSON lines on stdin/stdout.  print() is redirected to stderr."""
import json, sys

_out = sys.stdout
sys.stdout = sys.stderr
_in = sys.stdin


def _send(msg):
    _out.write(json.dumps(msg) + "\n"); _out.flush()


def _call(op, **kw):
    kw["op"] = op
    _send(kw)
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
    def __init__(self):
        info = _call("init")
        self.n_segments = info["n_segments"]
        self.b_min = info["b_min"]
        self.b_max = info["b_max"]
        self.t_overhead = info["t_ov"]
        self.throughput = info["throughput"]

    def run_segment(self, batch):
        return _call("run_segment", batch=float(batch))

    def elapsed(self):
        return _call("elapsed")["seconds"]
