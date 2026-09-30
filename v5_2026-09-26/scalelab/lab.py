"""Generic lab core: validation, cost accounting and noisy execution.

The loopback lab server, the oracle, the rivals and the literal-execution gate all go through
Session.run(), so what the gates certify is exactly what the agent can do.  Everything
lab-specific lives in `scalelab.labs.<spec["lab"]>` (default "pretrain").

spec (per task, JSON-serialisable):
  lab:     "pretrain" | "evallab" | "rllab" | "servelab"   (default "pretrain")
  knobs:   {name: {"type": "float", "min": a, "max": b} | {"type": "choice", "values": [...]}}
           each may carry "default"; a knob with a default may be omitted from a request
  fixed:   {cfg key: value}            values the agent cannot change
  caps:    {"run_cost": x, "total_cost": y, "max_runs": n}   (aliases: run_flops / total_flops)
  plus whatever the backend reads (see the backend module docstring)
"""
import copy, math
from . import labs


class LabError(Exception):
    pass


def _num(x, name):
    try:
        v = float(x)
    except Exception:
        raise LabError("%s must be a number, got %r" % (name, x))
    if not math.isfinite(v):
        raise LabError("%s must be finite" % name)
    return v


def caps_of(spec):
    c = spec["caps"]
    return {"run_cost": c.get("run_cost", c.get("run_flops")),
            "total_cost": c.get("total_cost", c.get("total_flops")),
            "max_runs": c["max_runs"]}


class Session:
    def __init__(self, params, spec, salt):
        self.bk = labs.backend(spec.get("lab", "pretrain"))
        self.p = self.bk.full(params); self.spec = spec; self.salt = salt
        self.caps = caps_of(spec)
        self.used = 0.0; self.n_runs = 0; self.log = []

    # v3 compatibility
    @property
    def used_flops(self):
        return self.used

    # ------------------------------------------------------------------ validation
    def validate(self, req):
        sp = self.spec; cfg = {}
        req = dict(req)
        seed = int(_num(req.pop("seed", 0), "seed"))
        if not 0 <= seed < 10 ** 6:
            raise LabError("seed must be an integer in [0, 1e6)")
        reserved = set(getattr(self.bk, "EXTRA_FIELDS", ()))
        for k in list(req):
            if k not in sp["knobs"] and k not in reserved:
                raise LabError("unknown or fixed knob %r (settable: %s)" % (k, ", ".join(sorted(sp["knobs"]))))
        for k, kd in sp["knobs"].items():
            if k not in req:
                if "default" in kd:
                    cfg[k] = kd["default"]; continue
                raise LabError("missing required knob %r" % k)
            v = req.pop(k)
            if kd["type"] == "float":
                v = _num(v, k)
                if not kd["min"] <= v <= kd["max"]:
                    raise LabError("%s=%g outside allowed range [%g, %g]" % (k, v, kd["min"], kd["max"]))
                if kd.get("int"):
                    if abs(v - round(v)) > 1e-9:
                        raise LabError("%s must be a whole number, got %g" % (k, v))
                    v = int(round(v))
            elif kd["type"] == "choice":
                if isinstance(kd["values"][0], (int, float)) and not isinstance(v, str):
                    v = _num(v, k)
                    if kd["values"] and isinstance(kd["values"][0], int):
                        v = int(round(v))
                    if v not in kd["values"]:
                        raise LabError("%s must be one of %s" % (k, kd["values"]))
                elif v not in kd["values"]:
                    raise LabError("%s must be one of %s" % (k, kd["values"]))
            cfg[k] = v
        cfg.update(sp.get("fixed", {}))
        self.bk.check(self, cfg)
        extra = self.bk.extras(self, req, cfg) if hasattr(self.bk, "extras") else {}
        return cfg, seed, extra

    # ------------------------------------------------------------------ cost
    def cost(self, cfg, extra=None):
        return float(self.bk.cost(self, cfg, extra or {}))

    # ------------------------------------------------------------------ execution
    def execute(self, cfg, seed, extra=None):
        return self.bk.execute(self, cfg, seed, extra or {})

    def mean(self, cfg, **kw):                     # pretrain blueprints use this directly
        return self.bk.mean(self, cfg, **kw)

    def diverges(self, cfg, seed):
        return self.bk.diverges(self, cfg, seed)

    def run(self, req):
        """Validate, check budget, execute, charge.  Raises LabError on any violation."""
        cfg, seed, extra = self.validate(req)
        caps = self.caps; c = self.cost(cfg, extra); unit = self.bk.COST_UNIT
        if c > caps["run_cost"] * (1 + 1e-9):
            raise LabError("request needs %.4g %s, above the per-request cap %.4g (%s)"
                           % (c, unit, caps["run_cost"], self.bk.COST_TEXT))
        if self.n_runs >= caps["max_runs"]:
            raise LabError("request limit reached (%d requests)" % caps["max_runs"])
        res = self.execute(cfg, seed, extra)
        charge = c * getattr(self.bk, "charge_factor", lambda r: 1.0)(res)
        if self.used + charge > caps["total_cost"] * (1 + 1e-9):
            raise LabError("insufficient budget: need %.4g %s, left %.4g" % (charge, unit, caps["total_cost"] - self.used))
        self.used += charge; self.n_runs += 1
        res["run_id"] = self.n_runs
        res["cost_charged"] = charge
        res["budget_left"] = caps["total_cost"] - self.used
        res["runs_left"] = caps["max_runs"] - self.n_runs
        if unit == "FLOPs":                        # v3 field names, kept for the pretraining tasks
            res["flops_charged"] = charge
            res["budget_left_flops"] = caps["total_cost"] - self.used
        self.log.append(copy.deepcopy(res))
        return res

    def status(self):
        caps = self.caps
        out = {"cost_unit": self.bk.COST_UNIT, "cost_used": self.used, "cost_left": caps["total_cost"] - self.used,
               "runs_used": self.n_runs, "runs_left": caps["max_runs"] - self.n_runs,
               "run_cost_cap": caps["run_cost"]}
        if self.bk.COST_UNIT == "FLOPs":
            out.update({"flops_used": self.used, "flops_left": caps["total_cost"] - self.used,
                        "run_flops_cap": caps["run_cost"]})
        return out
