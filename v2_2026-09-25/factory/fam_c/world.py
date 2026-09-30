"""Family C world: the structural model behind a fleet of historical training runs.

This file is authoring-only.  The agent image receives *data* (log.csv, queries.json, columns.md) and
never the generative code, so ground truth cannot be read off - it has to be recovered from the log.

Pathologies deliberately built into the historical assignment policy:
  * an unlogged hardware generation `hw_gen` drives both the operator's choice of micro_bs / seq_len and
    val_loss  -> naive regressions on knobs alone are confounded.  It is *fully blocked* by a pre-training
    hardware probe column (nccl_bw_gbps), which is a deterministic many-to-one function of hw_gen.
  * grad_norm_p95 is a during-run measurement on the causal path lr_scale -> val_loss: adjusting for it
    destroys part of the effect (over-adjustment).  throughput_toks_s is likewise post-treatment.
  * zero_stage and offload were always flipped together -> the single-knob contrast is not estimable.
  * opt_eps = 1e-6 was only ever used with seq_len = 1024 and interacts with seq_len -> the contrast the
    deployment asks about sits in an empty cell.
  * warmup = 2000 never occurs in the log at all, and warmup enters non-linearly.
  * lr_scale x warmup and offload x micro_bs interact -> a baseline-specific contrast cannot be read off
    an additive model.
"""
import numpy as np

LEVELS = {"lr_scale": [0.5, 1.0, 2.0], "warmup": [200, 1000], "micro_bs": [4, 8, 16],
          "grad_accum": [1, 2, 4], "zero_stage": [1, 2], "offload": [0, 1],
          "seq_len": [1024, 2048], "act_ckpt": [0, 1], "dropout": [0.0, 0.1],
          "opt_eps": [1e-8, 1e-6]}
KNOBS = list(LEVELS)
PROBE = ["nccl_bw_gbps", "sm_clock_mhz"]
DURING = ["grad_norm_p95", "throughput_toks_s", "step_time_ms"]

BASELINE = {"lr_scale": 1.0, "warmup": 1000, "micro_bs": 8, "grad_accum": 1, "zero_stage": 1,
            "offload": 0, "seq_len": 2048, "act_ckpt": 0, "dropout": 0.0, "opt_eps": 1e-8}

F = {"lr_scale": {0.5: 0.060, 1.0: 0.0, 2.0: -0.045},
     "warmup": {200: 0.030, 1000: 0.0, 2000: 0.110},
     "micro_bs": {4: 0.050, 8: 0.0, 16: -0.018},
     "grad_accum": {1: 0.0, 2: -0.022, 4: -0.030},
     "zero_stage": {1: 0.0, 2: 0.004},
     "offload": {0: 0.0, 1: 0.006},
     "seq_len": {1024: 0.0, 2048: -0.040},
     "act_ckpt": {0: 0.0, 1: 0.0},
     "dropout": {0.0: 0.0, 0.1: 0.015},
     "opt_eps": {1e-8: 0.0, 1e-6: -0.004}}
BASE_LOSS = 2.40
C_HW = -0.055
G_MED = 0.035            # coefficient of grad_norm_p95 in val_loss
SIG_Y = 0.0040
SIG_GN = 0.050


def inter(c):
    """Interaction part of the deterministic mean."""
    s = 0.0
    if c["lr_scale"] == 2.0 and c["warmup"] == 200:
        s += 0.070
    if c["opt_eps"] == 1e-6 and c["seq_len"] == 2048:
        s += 0.055
    if c["offload"] == 1 and c["micro_bs"] == 4:
        s += 0.020
    return s


def gn_det(c):
    """Deterministic, hardware-free part of grad_norm_p95 (the mediator)."""
    return 1.20 + 0.55 * (c["lr_scale"] - 1.0) + 0.010 * (c["micro_bs"] - 8)


def det_mean(c):
    """E[val_loss | config, hw_gen = 0], noise marginalised (mediator included)."""
    return BASE_LOSS + sum(F[k][c[k]] for k in KNOBS) + inter(c) + G_MED * gn_det(c)


def true_delta(knob, a, b, base):
    """Interventional contrast at a fixed baseline.  hw_gen and all noise are additive, so they cancel."""
    ca = dict(base); ca[knob] = a
    cb = dict(base); cb[knob] = b
    return det_mean(cb) - det_mean(ca)


BW = {0: [95.0, 100.0, 105.0], 1: [180.0, 190.0]}
CLK = {0: [1410, 1440], 1: [1755, 1800]}


def sample_log(n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        hw = int(rng.random() < 0.45)
        c = {}
        c["micro_bs"] = int(rng.choice([4, 8, 16], p=[0.10, 0.30, 0.60] if hw else [0.50, 0.35, 0.15]))
        c["seq_len"] = int(rng.choice([1024, 2048], p=[0.30, 0.70] if hw else [0.75, 0.25]))
        c["lr_scale"] = float(rng.choice([0.5, 1.0, 2.0], p=[0.25, 0.45, 0.30]))
        c["warmup"] = int(rng.choice([200, 1000], p=[0.40, 0.60]))
        c["grad_accum"] = int(rng.choice([1, 2, 4], p=[0.40, 0.35, 0.25]))
        z = int(rng.random() < 0.5)                       # zero_stage and offload are always flipped together
        c["zero_stage"] = 2 if z else 1
        c["offload"] = 1 if z else 0
        big = (c["micro_bs"] == 16) or (c["seq_len"] == 2048)
        c["act_ckpt"] = int(rng.random() < (0.80 if big else 0.15))
        c["dropout"] = float(rng.choice([0.0, 0.1], p=[0.60, 0.40]))
        c["opt_eps"] = 1e-6 if (c["seq_len"] == 1024 and rng.random() < 0.35) else 1e-8
        gn = gn_det(c) + 0.25 * hw + rng.normal(0, SIG_GN)
        y = BASE_LOSS + sum(F[k][c[k]] for k in KNOBS) + inter(c) + C_HW * hw \
            + G_MED * gn + rng.normal(0, SIG_Y)
        thr = 12000 * (1 + 0.35 * hw) * (c["micro_bs"] / 8.0) ** 0.5 / (1 + 0.25 * c["act_ckpt"]) \
            * (1024.0 / c["seq_len"]) ** 0.3 / (1 + 0.15 * c["offload"]) * (1 + rng.normal(0, 0.03))
        r = dict(c)
        r["nccl_bw_gbps"] = float(rng.choice(BW[hw]))
        r["sm_clock_mhz"] = int(rng.choice(CLK[hw]))
        r["grad_norm_p95"] = round(float(gn), 5)
        r["throughput_toks_s"] = round(float(thr), 1)
        r["step_time_ms"] = round(float(1000.0 * c["micro_bs"] * c["seq_len"] * c["grad_accum"] / thr), 3)
        r["val_loss"] = round(float(y), 6)
        rows.append(r)
    return rows


COLS = KNOBS + PROBE + DURING + ["val_loss"]


def queries():
    def q(i, knob, a, b, **ov):
        base = dict(BASELINE); base.update(ov)
        return {"id": "q%02d" % i, "knob": knob, "from": a, "to": b, "baseline": base}
    return [q(1, "lr_scale", 1.0, 2.0), q(2, "warmup", 200, 1000), q(3, "micro_bs", 8, 16),
            q(4, "grad_accum", 1, 2), q(5, "zero_stage", 1, 2), q(6, "offload", 0, 1),
            q(7, "seq_len", 1024, 2048), q(8, "act_ckpt", 0, 1), q(9, "dropout", 0.0, 0.1),
            q(10, "opt_eps", 1e-8, 1e-6), q(11, "warmup", 1000, 2000),
            q(12, "lr_scale", 1.0, 2.0, warmup=200)]
