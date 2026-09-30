"""ScaleLab lab core: validation, cost accounting and noisy execution.  The loopback lab server, the
oracle, the rivals and the literal-execution gate all go through Session.run(), so what the gates certify
is exactly what the agent can do.

spec (per task, JSON-serialisable):
  knobs:   {name: {"type": "float", "min": a, "max": b, "log": true} | {"type": "choice", "values": [...]}}
  fixed:   {cfg key: value}            values the agent cannot change
  caps:    {"run_flops": x, "total_flops": y, "max_runs": n}
  metrics: list among loss, qa, qa_clean, arith, arith_ll
  qa_items / qa_clean_items: benchmark sizes;  arith_weights: {k: w}
  checkpoints: bool  (C11 tasks: `ckpts` list of fractions and `cooldowns` list of fractions)
"""
import copy, math
import numpy as np
from . import world as W

DIV_CHARGE = 0.1   # a diverged run is detected early and charged 10% of its compute


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


class Session:
    def __init__(self, params, spec, salt):
        self.p = W.full(params); self.spec = spec; self.salt = salt
        self.used_flops = 0.0; self.n_runs = 0; self.log = []

    # ------------------------------------------------------------------ validation
    def validate(self, req):
        sp = self.spec; cfg = {}
        req = dict(req)
        seed = int(_num(req.pop("seed", 0), "seed"))
        if not 0 <= seed < 10 ** 6:
            raise LabError("seed must be an integer in [0, 1e6)")
        ckpts = req.pop("ckpts", None); cools = req.pop("cooldowns", None)
        for k in list(req):
            if k not in sp["knobs"]:
                raise LabError("unknown or fixed knob %r (settable: %s)" % (k, ", ".join(sorted(sp["knobs"]))))
        for k, kd in sp["knobs"].items():
            if k not in req:
                if "default" in kd:
                    cfg[k] = kd["default"]; continue
                raise LabError("missing required knob %r" % k)
            v = req[k]
            if kd["type"] == "float":
                v = _num(v, k)
                if not kd["min"] <= v <= kd["max"]:
                    raise LabError("%s=%g outside allowed range [%g, %g]" % (k, v, kd["min"], kd["max"]))
            elif kd["type"] == "choice":
                if isinstance(kd["values"][0], (int, float)) and not isinstance(v, str):
                    v = _num(v, k)
                    if v not in kd["values"]:
                        raise LabError("%s must be one of %s" % (k, kd["values"]))
                elif v not in kd["values"]:
                    raise LabError("%s must be one of %s" % (k, kd["values"]))
            cfg[k] = v
        cfg.update(sp.get("fixed", {}))
        if "pool" in cfg and "q" in cfg and cfg["pool"] > self.p["U0"] * (1 - cfg["q"]) * (1 + 1e-9):
            raise LabError("pool=%g exceeds the filtered corpus size %g at q=%g" % (cfg["pool"], self.p["U0"] * (1 - cfg["q"]), cfg["q"]))
        extra = {}
        if sp.get("checkpoints"):
            for nm, lst in (("ckpts", ckpts), ("cooldowns", cools)):
                if lst is None:
                    lst = []
                if not isinstance(lst, list) or len(lst) > 20:
                    raise LabError("%s must be a list of at most 20 fractions" % nm)
                vals = sorted(set(round(_num(x, nm), 6) for x in lst))
                if any(not 0.02 <= x <= 1.0 for x in vals):
                    raise LabError("%s fractions must lie in [0.02, 1]" % nm)
                extra[nm] = vals
            if extra["cooldowns"] and cfg.get("sched") != "wsd":
                raise LabError("cooldowns require sched='wsd'")
        elif ckpts is not None or cools is not None:
            raise LabError("this lab does not record intermediate checkpoints")
        return cfg, seed, extra

    # ------------------------------------------------------------------ cost
    def cost(self, cfg, extra=None):
        c = 6.0 * cfg["N"] * cfg["D"]
        if extra:
            c += sum(6.0 * cfg["N"] * self.p["cd"] * f * cfg["D"] for f in extra.get("cooldowns", []))
        return c

    # ------------------------------------------------------------------ execution
    def _wcfg(self, cfg):
        c = dict(cfg)
        if "sched" in c:
            c["sched"] = {"cosine": 0, "wsd": 1, "constant": 2}[c["sched"]]
        return c

    def mean(self, cfg, f=1.0, sched=None):
        c = self._wcfg(cfg); c["f"] = f
        if sched is not None:
            c["sched"] = sched
        return float(W.val_loss(self.p, c))

    def diverges(self, cfg, seed):
        p = self.p
        if not np.isfinite(p["h0"]):
            return False
        emax = float(W.eta_max(p, cfg["N"], cfg.get("wu", 0.01), cfg.get("qk", 0)))
        # per-run scatter of the edge; a spec may set div_jitter = 0 for a deterministic edge (then the
        # identification set of a censored quantity has crisp endpoints)
        jit = math.exp(self.spec.get("div_jitter", 0.04) * W.zdraw(self.salt, cfg, seed, "div"))
        return cfg["lr"] > emax * jit

    def execute(self, cfg, seed, extra=None):
        p = self.p; sp = self.spec; extra = extra or {}
        out = {"config": dict(cfg, seed=seed), "status": "ok"}
        if extra.get("ckpts") or extra.get("cooldowns"):
            out["config"].update({k: v for k, v in extra.items() if v})
        if self.diverges(cfg, seed):
            out["status"] = "diverged"; out["loss"] = None
            out["note"] = "loss spike and divergence detected early in training; run aborted"
            return out
        sg = float(W.sigma(p, cfg["N"]))
        zr = W.zdraw(self.salt, cfg, seed, "loss")
        wc = self._wcfg(cfg)
        mets = sp.get("metrics", ["loss"])
        if "loss" in mets:
            out["loss"] = round(float(W.val_loss(p, wc)) + sg * zr, 4)
        if "qa" in mets or "qa_clean" in mets:
            a, agg = W.qa_acc(p, wc)
            a = float(a); agg = float(agg)
            zs = 0.6 * zr + 0.8 * W.zdraw(self.salt, cfg, seed, "qa_seed")   # seed effect shared by both splits
            if "qa" in mets:
                n = sp["qa_items"]; s = math.sqrt(max(agg * (1 - agg), 1e-6) / n + (0.5 * sg) ** 2)
                out["qa_acc"] = round(min(1, max(0, agg + s * (0.5 * zs + 0.866 * W.zdraw(self.salt, cfg, seed, "qa")))), 4)
            if "qa_clean" in mets:
                n = sp["qa_clean_items"]; s = math.sqrt(max(a * (1 - a), 1e-6) / n + (0.5 * sg) ** 2)
                out["qa_clean_acc"] = round(min(1, max(0, a + s * (0.5 * zs + 0.866 * W.zdraw(self.salt, cfg, seed, "qac")))), 4)
        if "arith" in mets:
            w = sp["arith_weights"]; acc = float(W.exact_match(p, wc, w)); n = sp["arith_items"]
            s = math.sqrt(max(acc * (1 - acc), 1e-6) / n)
            out["arith_em"] = round(min(1, max(0, acc + s * W.zdraw(self.salt, cfg, seed, "arith"))), 4)
        if "arith_ll" in mets:
            ll = -float(W.answer_token_loss(p, wc))
            out["arith_ll"] = round(ll + 0.6 * sg * W.zdraw(self.salt, cfg, seed, "arith_ll") + 0.4 * sg * zr, 4)
        if extra.get("ckpts"):
            out["checkpoints"] = []
            for f in extra["ckpts"]:
                m = self.mean(cfg, f=f)
                z = 0.8 * zr + 0.6 * W.zdraw(self.salt, cfg, seed, "ck%.6f" % f)
                out["checkpoints"].append({"frac": f, "tokens": f * cfg["D"], "loss": round(m + sg * z, 4)})
        if extra.get("cooldowns"):
            out["cooldown_branches"] = []
            for f in extra["cooldowns"]:
                sub = dict(cfg, D=f * cfg["D"])          # a cooled-down branch is a finished WSD run of length f*D
                m = self.mean(sub, f=1.0)
                z = 0.8 * zr + 0.6 * W.zdraw(self.salt, cfg, seed, "cd%.6f" % f)
                out["cooldown_branches"].append({"frac": f, "tokens": f * cfg["D"], "loss": round(m + sg * z, 4)})
        return out

    def run(self, req):
        """Validate, check budget, execute, charge.  Raises LabError on any violation."""
        cfg, seed, extra = self.validate(req)
        caps = self.spec["caps"]; c = self.cost(cfg, extra)
        if c > caps["run_flops"] * (1 + 1e-9):
            raise LabError("run needs %.3g FLOPs, above the per-run cap %.3g (FLOPs = 6*N*D%s)" %
                           (c, caps["run_flops"], " + cooldown branches" if extra.get("cooldowns") else ""))
        if self.n_runs >= caps["max_runs"]:
            raise LabError("run limit reached (%d runs)" % caps["max_runs"])
        res = self.execute(cfg, seed, extra)
        charge = c * (DIV_CHARGE if res["status"] == "diverged" else 1.0)
        if self.used_flops + charge > caps["total_flops"] * (1 + 1e-9):
            raise LabError("insufficient compute budget: need %.3g, left %.3g" % (charge, caps["total_flops"] - self.used_flops))
        self.used_flops += charge; self.n_runs += 1
        res["run_id"] = self.n_runs; res["flops_charged"] = charge
        res["budget_left_flops"] = caps["total_flops"] - self.used_flops; res["runs_left"] = caps["max_runs"] - self.n_runs
        self.log.append(copy.deepcopy(res))
        return res

    def status(self):
        caps = self.spec["caps"]
        return {"flops_used": self.used_flops, "flops_left": caps["total_flops"] - self.used_flops,
                "runs_used": self.n_runs, "runs_left": caps["max_runs"] - self.n_runs,
                "run_flops_cap": caps["run_flops"]}
