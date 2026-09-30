"""Input generators for family O (shared by generator, verifier and difficulty mining)."""
import numpy as np

PROFILES = ["mix", "zeros", "oversize", "tiny", "bimodal", "desc", "exact", "wide"]


def make_lengths(profile, n, C, seed):
    r = np.random.default_rng(seed)
    if profile == "mix":
        x = np.clip(np.round(r.lognormal(np.log(C * 0.22), 0.9, n)), 1, int(C * 1.4))
    elif profile == "zeros":
        x = np.clip(np.round(r.lognormal(np.log(C * 0.3), 0.8, n)), 0, C)
        x[r.random(n) < 0.18] = 0
    elif profile == "oversize":
        x = np.clip(np.round(r.lognormal(np.log(C * 0.5), 1.0, n)), 0, int(C * 3))
    elif profile == "tiny":
        x = r.integers(1, max(2, C // 24), n)
    elif profile == "bimodal":
        x = np.where(r.random(n) < 0.5, r.integers(1, max(2, C // 20), n),
                     r.integers(int(C * 0.55), C + 1, n))
    elif profile == "desc":
        x = np.sort(np.clip(np.round(r.lognormal(np.log(C * 0.3), 0.9, n)), 0, int(C * 1.2)))[::-1]
    elif profile == "exact":
        base = np.array([C, C - 1, 0, 1, C // 2, C // 2 + 1, C + 1, C - 2])
        x = base[r.integers(0, len(base), n)]
    elif profile == "wide":
        x = r.integers(0, int(C * 1.3), n)
    else:
        raise ValueError(profile)
    return [int(v) for v in x]
