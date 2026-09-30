"""Linear warmup followed by cosine decay to min_lr_ratio * lr."""
import math


def lr_at(step, cfg):
    base = cfg["lr"]
    warm = cfg["warmup_steps"]
    total = cfg["steps"]
    if step < warm:
        return base * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    mr = cfg["min_lr_ratio"]
    return base * (mr + (1 - mr) * 0.5 * (1 + math.cos(math.pi * t)))
