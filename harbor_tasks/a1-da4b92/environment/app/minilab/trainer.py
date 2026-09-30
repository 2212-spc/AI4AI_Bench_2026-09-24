"""Training loop (v2: gradient accumulation + clipping refactor)."""
import numpy as np

from .model import init_params, loss_and_grads
from .optim import SGD
from .schedule import lr_at


def _global_norm(grads):
    return float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))


def train(cfg, x_train, y_train, seed):
    """Train one model; returns the parameter dict {W1, b1, W2, b2}."""
    rng = np.random.default_rng(seed)
    params = init_params(rng, x_train.shape[1], cfg["hidden"])
    opt = SGD(params, lr=cfg["lr"], momentum=cfg["momentum"],
              weight_decay=cfg["weight_decay"])
    k = cfg["grad_accum_steps"]
    mb = cfg["micro_batch_size"]
    n = x_train.shape[0]
    clip = cfg.get("clip_norm")

    for step in range(cfg["steps"]):
        acc = {name: np.zeros_like(v) for name, v in params.items()}
        for _ in range(k):
            idx = rng.integers(0, n, size=mb)
            _, g = loss_and_grads(params, x_train[idx], y_train[idx], denom=len(idx))
            for name in acc:
                acc[name] += g[name]
        if clip is not None:
            norm = _global_norm(acc)
            if norm > clip:
                for name in acc:
                    acc[name] *= clip / norm
        opt.lr = lr_at(step, cfg)
        opt.step(acc)
        if not np.isfinite(params["W2"]).all():
            raise FloatingPointError("training diverged at step %d" % step)
    return params
