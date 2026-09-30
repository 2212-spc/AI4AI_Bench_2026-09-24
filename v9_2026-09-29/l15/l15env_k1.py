"""Client library for K1 estimators (mounted read-only as /lib/l15env.py inside the estimator sandbox).

    env.rows              list of {"id", "topic", "length", "judge_win"} - the whole arena, free
    env.label_budget      how many human labels you may buy on this arena
    env.topics            the topic values
    env.lengths           the length values
    env.target_coverage   nominal coverage of the interval you are asked for
    env.production_mix    {"topic|length": weight} - the mix you are estimating on
    env.label(ids)        -> {"<id>": True/False}; True = the candidate really won
    env.submit(point, lo, hi)   report and return; call exactly once

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
        self.rows = info["rows"]
        self.label_budget = info["label_budget"]
        self.topics = info["topics"]
        self.lengths = info["lengths"]
        self.target_coverage = info["target_coverage"]
        self.production_mix = info["production_mix"]

    def label(self, ids):
        return _call("label", ids=[int(i) for i in ids])["labels"]

    def submit(self, point, lo, hi):
        return _call("submit", point=float(point), lo=float(lo), hi=float(hi))
