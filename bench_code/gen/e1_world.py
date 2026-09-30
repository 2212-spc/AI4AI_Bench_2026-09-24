"""E1 world: 6 candidate changes on top of a correct MiniLab baseline."""
import sys; sys.path.insert(0, "/tmp/bench/lab")
import numpy as np
import engine as E

NAMES = ["c1", "c2", "c3", "c4", "c5", "c6"]


def cell_cfg(base, deltas, mask):
    cfg = dict(base); sep_rng = False
    for i, nm in enumerate(NAMES):
        if mask >> i & 1:
            d = deltas[nm]
            if nm == "c6": sep_rng = True
            else: cfg.update(d)
    return cfg, sep_rng


def _split(cfg):
    flags = {k[5:]: v for k, v in cfg.items() if k.startswith("flag_")}
    return {k: v for k, v in cfg.items() if not k.startswith("flag_")}, flags


def run_cell(world, mask, seed):
    """returns val mse (diverged -> trivial-predictor mse)"""
    cfg, sep = cell_cfg(world["base"], world["deltas"], mask)
    xtr, ytr, xva, yva = world["data"]
    cfg, flags = _split(cfg)
    p = E.train(cfg, xtr, ytr, seed, sep_rng=sep, **flags)
    if p is None: return float(np.var(yva)), True
    v = E.mse(p, xva, yva)
    if not np.isfinite(v) or v > np.var(yva): return float(np.var(yva)), True
    return v, False


def make_world(teacher_seed, base, deltas, world_kw, data_seed):
    T = E.make_teacher(teacher_seed, **world_kw)
    xtr, ytr = E.sample(T, 20000, data_seed); xva, yva = E.sample(T, 4000, data_seed + 1)
    return {"base": base, "deltas": deltas, "data": (xtr, ytr, xva, yva)}


def build_world(params):
    base = dict(hidden=96, lr=params["lr"], momentum=params["mom"], weight_decay=0.0, warmup_steps=params["warm"], steps=400,
                min_lr_ratio=0.05, grad_accum_steps=1, micro_batch_size=32, clip_norm=None)
    deltas = {"c1": {"lr": params["lr"] * params["lr_f"]}, "c2": {"momentum": params["mom2"]}, "c3": {"clip_norm": params["clip"]},
              "c4": {"hidden": params.get("hid2", 160)}, "c5": {"weight_decay": params.get("wd2", 3e-4)}, "c6": {}}
    for k, v in params.get("override", {}).items(): deltas[k] = v
    return make_world(params["tseed"], base, deltas, dict(d_in=32, t_hidden=64, noise=0.1), params["dseed"])


def mask_of(names):
    m = 0
    for n in names: m |= 1 << NAMES.index(n)
    return m


def names_of(mask):
    return [n for i, n in enumerate(NAMES) if mask >> i & 1]
