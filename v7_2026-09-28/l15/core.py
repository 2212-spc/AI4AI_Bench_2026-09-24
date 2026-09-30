"""L1.5 lab core: budgeted, replayable sessions over a hidden world.

A task module (l15/tasks/<name>.py) defines a subclass of `World`:
  * `OPS`      : {op_name: (cost_fn(world, args) -> float, run_fn(world, args, ctx) -> dict, doc)}
  * `public_spec()` : what `lab spec` shows (ops, knobs, ranges, costs, budget) - never hidden params
  * `grade(art_dir, ledger)` : score dict from the declared artifacts only (+ ledger for integrity checks)
  * `ARTIFACTS` : list of file names (relative to /app) the verifier may see

Determinism: every op draws its randomness from rng(salt, call_index), so a session is fully described by
its ledger and can be rebuilt after the host kills the lab process (which happens every tool call).
"""
import hashlib, json, math, os, time
import numpy as np


class LabError(Exception):
    pass


def rng_for(salt, *keys):
    h = hashlib.sha256(("%s|" % salt + "|".join(str(k) for k in keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def num(x, name, lo=None, hi=None, integer=False):
    try:
        v = float(x)
    except Exception:
        raise LabError("%s must be a number, got %r" % (name, x))
    if not math.isfinite(v):
        raise LabError("%s must be finite" % name)
    if integer:
        if abs(v - round(v)) > 1e-9:
            raise LabError("%s must be an integer" % name)
        v = int(round(v))
    if lo is not None and v < lo:
        raise LabError("%s=%g is below the allowed minimum %g" % (name, v, lo))
    if hi is not None and v > hi:
        raise LabError("%s=%g is above the allowed maximum %g" % (name, v, hi))
    return v


def choice(x, name, opts):
    if x not in opts:
        raise LabError("%s must be one of %s, got %r" % (name, list(opts), x))
    return x


class World:
    NAME = "base"
    ARTIFACTS = []
    OPS = {}
    BUDGET_UNIT = "credits"

    def __init__(self, cfg):
        self.cfg = cfg                  # full hidden instance: {"task","params","salt","budget",...}
        self.p = cfg["params"]
        self.salt = cfg["salt"]
        self.budget = float(cfg["budget"])

    def public_spec(self):
        raise NotImplementedError

    def grade(self, art_dir, ledger):
        raise NotImplementedError


class Session:
    """Budget + ledger for one run.  `ledger_path` is outside the sandbox (host side of the run dir)."""

    def __init__(self, world, ledger_path=None, app_dir=None):
        self.w = world
        self.ledger_path = ledger_path
        self.app_dir = app_dir
        self.spent = 0.0
        self.n = 0
        self.records = []
        if ledger_path and os.path.exists(ledger_path):
            for line in open(ledger_path):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                self.records.append(r)
                self.spent += r["cost"]
                self.n = max(self.n, r["i"] + 1)

    def left(self):
        return self.w.budget - self.spent

    def call(self, op, args):
        if op == "spec":
            s = self.w.public_spec()
            s["budget_total"] = self.w.budget
            s["budget_left"] = round(self.left(), 6)
            return s
        if op == "status":
            return {"budget_total": self.w.budget, "budget_spent": round(self.spent, 6),
                    "budget_left": round(self.left(), 6), "n_calls": self.n}
        if op == "history":
            k = int(args.get("last", 20)) if isinstance(args, dict) else 20
            return {"history": [{"i": r["i"], "op": r["op"], "args": r["args"], "cost": r["cost"],
                                 "result": r.get("result")} for r in self.records[-k:]]}
        if op not in self.w.OPS:
            raise LabError("unknown op %r; available: %s" % (op, sorted(list(self.w.OPS) + ["spec", "status", "history"])))
        cost_fn, run_fn, _doc = self.w.OPS[op]
        args = dict(args or {})
        cost = float(cost_fn(self.w, args))
        if cost > self.left() + 1e-9:
            raise LabError("insufficient budget: this call costs %.6g %s, %.6g left" % (cost, self.w.BUDGET_UNIT, self.left()))
        ctx = {"rng": rng_for(self.w.salt, "call", self.n), "i": self.n, "app_dir": self.app_dir, "session": self}
        res = run_fn(self.w, args, ctx)
        rec = {"i": self.n, "t": time.time(), "op": op, "args": args, "cost": cost, "result": res}
        self.n += 1
        self.spent += cost
        self.records.append(rec)
        if self.ledger_path:
            with open(self.ledger_path, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
        out = dict(res)
        out["call_id"] = rec["i"]
        out["cost"] = round(cost, 6)
        out["budget_left"] = round(self.left(), 6)
        return out


def load_world(task_dir):
    cfg = json.load(open(os.path.join(task_dir, "hidden", "world.json")))
    return make_world(cfg)


def make_world(cfg):
    import importlib
    mod = importlib.import_module("l15.tasks." + cfg["task"])
    return mod.World(cfg)


def read_json_artifact(art_dir, name):
    p = os.path.join(art_dir, name)
    if not os.path.exists(p):
        return None, "missing %s" % name
    try:
        return json.load(open(p)), None
    except Exception as e:
        return None, "unparseable %s: %s" % (name, type(e).__name__)


def interval_ok(iv, truth, max_rel_width=None, max_abs_width=None):
    """Mechanical check of a submitted {"lo","hi"} interval."""
    try:
        lo, hi = float(iv["lo"]), float(iv["hi"])
    except Exception:
        return False, "malformed interval"
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        return False, "malformed interval"
    if max_abs_width is not None and hi - lo > max_abs_width:
        return False, "interval wider than allowed (%.4g > %.4g)" % (hi - lo, max_abs_width)
    if max_rel_width is not None and truth != 0 and (hi - lo) / abs(truth) > max_rel_width:
        return False, "interval wider than allowed"
    return (lo <= truth <= hi), ("covers" if lo <= truth <= hi else "misses")
