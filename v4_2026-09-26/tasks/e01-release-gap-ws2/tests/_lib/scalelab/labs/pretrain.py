"""Pretraining lab (ScaleLab v3 behaviour, moved behind the backend protocol).

spec extras:  metrics (loss|qa|qa_clean|arith|arith_ll), qa_items, qa_clean_items, arith_weights,
              arith_items, checkpoints (bool), div_jitter.
"""
import math
import numpy as np
from .. import world as W

NAME = "pretrain"
COST_UNIT = "FLOPs"
COST_TEXT = "FLOPs = 6*N*D (a run is charged this even if you only look at its end)"
DIV_CHARGE = 0.1   # a diverged run is detected early and charged 10% of its compute
BASE = W.BASE
# request fields that are not knobs: `extras` consumes them, so `validate` must not reject them
EXTRA_FIELDS = ("ckpts", "cooldowns")


def full(p):
    return W.full(p)


def check(sess, cfg):
    from ..lab import LabError
    if "pool" in cfg and "q" in cfg and cfg["pool"] > sess.p["U0"] * (1 - cfg["q"]) * (1 + 1e-9):
        raise LabError("pool=%g exceeds the filtered corpus size %g at q=%g"
                       % (cfg["pool"], sess.p["U0"] * (1 - cfg["q"]), cfg["q"]))


def extras(sess, req, cfg):
    from ..lab import LabError, _num
    ckpts = req.pop("ckpts", None); cools = req.pop("cooldowns", None)
    extra = {}
    if sess.spec.get("checkpoints"):
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
    return extra


def cost(sess, cfg, extra):
    c = 6.0 * cfg["N"] * cfg["D"]
    if extra:
        c += sum(6.0 * cfg["N"] * sess.p["cd"] * f * cfg["D"] for f in extra.get("cooldowns", []))
    return c


def charge_factor(res):
    return DIV_CHARGE if res.get("status") == "diverged" else 1.0


def _wcfg(cfg):
    c = dict(cfg)
    if "sched" in c:
        c["sched"] = {"cosine": 0, "wsd": 1, "constant": 2}[c["sched"]]
    return c


def mean(sess, cfg, f=1.0, sched=None):
    c = _wcfg(cfg); c["f"] = f
    if sched is not None:
        c["sched"] = sched
    return float(W.val_loss(sess.p, c))


def diverges(sess, cfg, seed):
    p = sess.p
    if not np.isfinite(p["h0"]):
        return False
    emax = float(W.eta_max(p, cfg["N"], cfg.get("wu", 0.01), cfg.get("qk", 0)))
    jit = math.exp(sess.spec.get("div_jitter", 0.04) * W.zdraw(sess.salt, cfg, seed, "div"))
    return cfg["lr"] > emax * jit


def execute(sess, cfg, seed, extra=None):
    p = sess.p; sp = sess.spec; extra = extra or {}
    out = {"config": dict(cfg, seed=seed), "status": "ok"}
    if extra.get("ckpts") or extra.get("cooldowns"):
        out["config"].update({k: v for k, v in extra.items() if v})
    if diverges(sess, cfg, seed):
        out["status"] = "diverged"; out["loss"] = None
        out["note"] = "loss spike and divergence detected early in training; run aborted"
        return out
    sg = float(W.sigma(p, cfg["N"]))
    zr = W.zdraw(sess.salt, cfg, seed, "loss")
    wc = _wcfg(cfg)
    mets = sp.get("metrics", ["loss"])
    if "loss" in mets:
        out["loss"] = round(float(W.val_loss(p, wc)) + sg * zr, 4)
    if "qa" in mets or "qa_clean" in mets:
        a, agg = W.qa_acc(p, wc)
        a = float(a); agg = float(agg)
        zs = 0.6 * zr + 0.8 * W.zdraw(sess.salt, cfg, seed, "qa_seed")   # seed effect shared by both splits
        if "qa" in mets:
            n = sp["qa_items"]; s = math.sqrt(max(agg * (1 - agg), 1e-6) / n + (0.5 * sg) ** 2)
            out["qa_acc"] = round(min(1, max(0, agg + s * (0.5 * zs + 0.866 * W.zdraw(sess.salt, cfg, seed, "qa")))), 4)
        if "qa_clean" in mets:
            n = sp["qa_clean_items"]; s = math.sqrt(max(a * (1 - a), 1e-6) / n + (0.5 * sg) ** 2)
            out["qa_clean_acc"] = round(min(1, max(0, a + s * (0.5 * zs + 0.866 * W.zdraw(sess.salt, cfg, seed, "qac")))), 4)
    if "arith" in mets:
        w = sp["arith_weights"]; acc = float(W.exact_match(p, wc, w)); n = sp["arith_items"]
        s = math.sqrt(max(acc * (1 - acc), 1e-6) / n)
        out["arith_em"] = round(min(1, max(0, acc + s * W.zdraw(sess.salt, cfg, seed, "arith"))), 4)
    if "arith_ll" in mets:
        ll = -float(W.answer_token_loss(p, wc))
        out["arith_ll"] = round(ll + 0.6 * sg * W.zdraw(sess.salt, cfg, seed, "arith_ll") + 0.4 * sg * zr, 4)
    if extra.get("ckpts"):
        out["checkpoints"] = []
        for f in extra["ckpts"]:
            m = mean(sess, cfg, f=f)
            z = 0.8 * zr + 0.6 * W.zdraw(sess.salt, cfg, seed, "ck%.6f" % f)
            out["checkpoints"].append({"frac": f, "tokens": f * cfg["D"], "loss": round(m + sg * z, 4)})
    if extra.get("cooldowns"):
        out["cooldown_branches"] = []
        for f in extra["cooldowns"]:
            sub = dict(cfg, D=f * cfg["D"])          # a cooled-down branch is a finished WSD run of length f*D
            m = mean(sess, sub, f=1.0)
            z = 0.8 * zr + 0.6 * W.zdraw(sess.salt, cfg, seed, "cd%.6f" % f)
            out["cooldown_branches"].append({"frac": f, "tokens": f * cfg["D"], "loss": round(m + sg * z, 4)})
    return out


MANUAL = r"""## 1. What the simulator models (abstraction boundary)

ScaleLab is a **simulated** language-model pretraining lab.  You cannot train real models here; every
`lab run` is answered by a simulator.  The simulator is not a replica of any real training stack.  Its
mechanisms are modelled on regularities reported in the scaling-law and training-dynamics literature,
but **its constants were drawn fresh for this lab**.  Published numbers (Chinchilla exponents,
"optimal" learning rates, epoch rules of thumb, and so on) describe other worlds.  Measure; do not recall.

Quantities: N = non-embedding parameters; D = training tokens; B = batch size in tokens;
lr = peak learning rate; FLOPs = 6*N*D (a run is charged this even if you only look at its end).
`loss` is the final validation loss in nats/token on a fixed held-out web set.

The simulator contains the following kinds of effects.  **Any given lab may switch some of them off
(held neutral); which ones are active is not stated.**

- *Size and data.*  Loss falls with N and with the effective number of tokens, with diminishing returns, towards an irreducible floor.
- *Learning rate.*  Loss rises on both sides of an optimal peak learning rate; the rise need not be symmetric.  The optimum may move with N, D and B.
- *Batch size.*  Beyond a critical batch size, larger batches waste tokens; the critical batch size may move with the data horizon.
- *Stability.*  Above an edge learning rate a run diverges.  The edge may move with N, warmup and qk-layernorm.  Divergence is detected early; a diverged run is charged 10% of its FLOPs.
- *Weight decay.*  AdamW's averaging timescale B/(lr*wd*D) has an optimum that may depend on tokens per parameter.
- *Repetition.*  Repeating unique tokens (epochs) gives diminishing value.
- *Quality filtering.*  Filtering can raise the value of each token but shrinks the pool of unique tokens.
- *Metrics.*  Task metrics (accuracy, exact match) are functions of the model's loss and of the metric's definition; some are not linear in loss.
- *Contamination.*  Some data sources can contain benchmark items; memorisation can grow with scale and exposure.
- *Schedules.*  An intermediate checkpoint of a long run is not the same as a finished short run; the gap depends on the learning-rate schedule.
- *Seed noise.*  Every metric has seed noise; its size can depend on N.

Not modelled: architecture beyond N, tokenizer, hardware, data order, wall-clock time.

## 2. Guarantees

- **Scale consistency.**  Every mechanism is a fixed law whose constants do not depend on scale.
  Relationships you measure in the lab continue, by the same law and with the same constants, to the
  production scales named in the questions.  No effect switches on or off outside the lab's range,
  and there are no hidden thresholds, **except** where Section 5 lists a *known unknown*.
- **Determinism.**  Re-running an identical configuration with the same seed returns the identical
  numbers (and is charged again).  To average out noise, use different seeds.
- **Notebook runs are real lab runs** (seed 0 unless stated), produced by the same simulator you query.
  The lab notes that accompany them are the team's interpretation and may be wrong.
"""

CLI_HELP = r"""## 4. Using the lab

The `lab` command talks to the lab service (no other network access is needed or available).

    lab spec                                  # knobs, fixed settings, caps, metrics (free)
    lab run N=6e7 D=1.5e10 lr=0.002 seed=3    # one training run; prints a JSON result
    lab batch plan.json                       # a JSON list of runs, e.g. [{"N":6e7,"D":1.5e10,"lr":0.002,"seed":3}, ...]
    lab status                                # compute used / left (free)
    lab history                               # every run you have made (free)

Settable knobs must all be given (unless the spec lists a default); fixed settings cannot be changed.
A run above the per-run FLOP cap, beyond the run limit, or beyond the remaining budget is refused and
not charged.  `lab batch` stops at the first refused run.  Results are also appended to
`/app/lab_runs.jsonl`.
"""
