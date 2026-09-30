"""Family S world: a *designed* sweep whose results table is silently incomplete.

Family C tested confounding in an observational log.  This world removes confounding entirely - the sweep
is a replicated factorial and assignment is by design - and replaces it with the pathology that actually
eats production sweeps: the rows you never see.

  * the launcher aborts a run whose final validation loss exceeds a guard tau, and an aborted run is
    never written to results.csv.  The censoring depends on the *outcome*, so every complete-case
    statistic in a high-loss region is biased downwards, and the bias is largest exactly where the
    interesting effects live.
  * the cluster also preempts low-priority jobs.  Preemption depends on `batch` only, never on the
    outcome, so it is harmless for cell means - but it drills conspicuous holes that invite
    over-correction, and the two causes of a missing row cannot be told apart run by run.
  * one region (lr=1e-3 with init_scale=1.0 and sched='cosine') sits so far above the guard that not a
    single launched run survives.  Its cell mean is bounded below and nothing else: any answer that
    reports a number there is extrapolating from an assumed functional form, not reading the data.
  * precision='fp8' was only ever launched at batch=1024 and precision='fp16' was never launched at all,
    so "underdetermined" is not a synonym for "censored".

Authoring-only.  The agent image receives manifest.csv, results.csv, launcher.md and queries.json.
"""
import math
import numpy as np

LEVELS = {"lr": [1e-4, 3e-4, 1e-3], "batch": [256, 512, 1024], "sched": ["cosine", "wsd"],
          "init_scale": [0.5, 1.0], "precision": ["bf16", "fp8", "fp16"]}
KNOBS = list(LEVELS)
BASELINE = {"lr": 3e-4, "batch": 512, "sched": "cosine", "init_scale": 0.5, "precision": "bf16"}

TAU = 2.955                      # divergence guard on the final validation loss
BASE = 2.62
SIG = 0.09                       # run-to-run noise on val_loss
REPS = 2000

MAIN = {"lr": {1e-4: 0.085, 3e-4: 0.0, 1e-3: 0.310},
        "batch": {256: 0.030, 512: 0.0, 1024: -0.012},
        "sched": {"cosine": 0.0, "wsd": -0.035},
        "init_scale": {0.5: 0.0, 1.0: 0.022},
        "precision": {"bf16": 0.0, "fp8": 0.018, "fp16": 0.0}}

# A large initialisation scale is mildly bad on its own and catastrophic at the largest learning rate;
# the wsd schedule tames that instability; small batches hurt more at tiny learning rates.
INTER = [(lambda c: c["lr"] == 1e-3 and c["init_scale"] == 1.0, 0.400),
         (lambda c: c["lr"] == 1e-3 and c["sched"] == "wsd", -0.330),
         (lambda c: c["batch"] == 256 and c["lr"] == 1e-4, 0.040)]

# The three-way term lives entirely inside the region where no run survives, so the data cannot see it.
# `free_term` is the value that an indistinguishable alternative world is free to choose.
def three_way(c, free_term=0.350):
    return free_term if (c["lr"] == 1e-3 and c["init_scale"] == 1.0 and c["sched"] == "cosine") else 0.0


PREEMPT = {256: 0.05, 512: 0.10, 1024: 0.28}      # queue pressure, independent of the outcome


def det_mean(c, free_term=0.350):
    return BASE + sum(MAIN[k][c[k]] for k in KNOBS) + sum(v for f, v in INTER if f(c)) \
        + three_way(c, free_term)


def true_delta(knob, a, b, base, free_term=0.350):
    ca = dict(base); ca[knob] = a
    cb = dict(base); cb[knob] = b
    return det_mean(cb, free_term) - det_mean(ca, free_term)


def allowed(c):
    """Cells the sweep was allowed to launch."""
    if c["precision"] == "fp16":
        return False
    if c["precision"] == "fp8" and c["batch"] != 1024:
        return False
    return True


def cells():
    out = []
    for lr in LEVELS["lr"]:
        for b in LEVELS["batch"]:
            for s in LEVELS["sched"]:
                for i in LEVELS["init_scale"]:
                    for p in LEVELS["precision"]:
                        c = {"lr": lr, "batch": b, "sched": s, "init_scale": i, "precision": p}
                        if allowed(c):
                            out.append(c)
    return out


def key(c):
    return tuple(c[k] for k in KNOBS)


def p_guard(c, free_term=0.350):
    """True fraction of launched runs in a cell whose final loss exceeds the guard."""
    return float(0.5 * math.erfc((TAU - det_mean(c, free_term)) / (SIG * math.sqrt(2.0))))


def p_survive(c, free_term=0.350):
    """Per-run probability that a launched run in cell c is *not* removed by the guard."""
    return float(0.5 * math.erfc((det_mean(c, free_term) - TAU) / (SIG * math.sqrt(2.0))))


def sample(seed):
    """(manifest rows, result rows).  A manifest row exists for every launched run; a result row exists
    only for runs that were neither preempted nor aborted by the guard."""
    rng = np.random.default_rng(seed)
    man, res = [], []
    rid = 100000
    for c in cells():
        mu = det_mean(c)
        for _ in range(REPS):
            rid += 1
            y = float(mu + rng.normal(0, SIG))
            row = dict(c); row["run_id"] = rid
            man.append(row)
            if rng.random() < PREEMPT[c["batch"]] or y > TAU:
                continue
            r = dict(row)
            r["steps"] = 20000
            r["tokens_seen"] = int(c["batch"]) * 2048 * 20000
            r["val_loss"] = round(y, 6)
            r["status"] = "ok"
            res.append(r)
    return man, res


MAN_COLS = ["run_id"] + KNOBS
RES_COLS = ["run_id"] + KNOBS + ["steps", "tokens_seen", "status", "val_loss"]


def queries():
    def q(i, knob, a, b, **ov):
        base = dict(BASELINE); base.update(ov)
        return {"id": "q%02d" % i, "knob": knob, "from": a, "to": b, "baseline": base}
    return [
        q(1, "lr", 3e-4, 1e-4),                                       # clean
        q(2, "sched", "cosine", "wsd"),                               # clean
        q(3, "lr", 3e-4, 1e-3),                                       # target cell ~29% censored
        q(4, "init_scale", 0.5, 1.0, lr=1e-3, sched="wsd"),           # target cell ~53% censored
        q(5, "batch", 512, 1024),                                     # heavy preemption, zero censoring
        q(6, "init_scale", 0.5, 1.0),                                 # clean
        q(7, "lr", 3e-4, 1e-3, init_scale=1.0),                       # target cell has no survivor
        q(8, "precision", "bf16", "fp8"),                             # cell never launched at batch=512
        q(9, "precision", "bf16", "fp16", batch=1024),                # level never launched
        q(10, "batch", 512, 1024, lr=1e-3, sched="wsd", init_scale=1.0),  # both censored, very
        #                                                             different preemption, tiny true effect
        q(11, "batch", 512, 256, lr=1e-4),                            # clean, interaction present
        q(12, "lr", 1e-4, 1e-3, sched="wsd", init_scale=1.0, batch=1024),  # one clean, one censored cell
        q(13, "sched", "cosine", "wsd", lr=1e-3, init_scale=1.0),     # one dead cell, one censored
    ]


def report_cells():
    """Cells whose guard-removal fraction the submission must report."""
    def c(**ov):
        d = dict(BASELINE); d.update(ov); return d
    return [("r1", c()),
            ("r2", c(lr=1e-3)),
            ("r3", c(lr=1e-3, init_scale=1.0, sched="wsd")),
            ("r4", c(batch=1024)),
            ("r5", c(lr=1e-4, batch=256)),
            ("r6", c(lr=1e-3, init_scale=1.0))]
