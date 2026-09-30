"""Hidden world for sched_law: exact expected loss of SGD on a noisy quadratic model (NQM).

Per eigen-direction i with curvature h_i:  v_i <- (1 - lr*h_i)^2 v_i + lr^2 * s_i
loss(t) = L_inf + 0.5 * sum_i h_i v_i(t).   (Zhang et al. 2019 NQM, expected dynamics.)
The agent never sees this file; it sees only noisy loss curves.
"""
import numpy as np

T_TRAIN = 2000
EVAL_EVERY = 20


def make_world(seed, gamma, sigma2, K=64, h_lo=1e-6, L_inf=None, init_excess=None):
    """Log-uniform curvature spectrum h in [h_lo, 1]; per-mode initial energy h*c ~ h^gamma
    gives power-law bias decay ~ (sum lr)^-gamma; noise cov s = sigma2*h gives an lr-proportional floor."""
    rng = np.random.default_rng(seed)
    h = np.logspace(0, np.log10(h_lo), K) * np.exp(rng.normal(0, 0.08, K))
    e = h ** gamma * np.exp(rng.normal(0, 0.15, K))              # 0.5*h*c per mode
    tot = init_excess if init_excess is not None else rng.uniform(3.0, 4.5)
    e = e / e.sum() * tot
    c = 2 * e / h
    s = sigma2 * h
    if L_inf is None:
        L_inf = rng.uniform(1.8, 2.6)
    return dict(h=h, c=c, s=s, L_inf=float(L_inf), seed=seed, gamma=gamma, sigma2=sigma2)


def run(world, lrs, every=EVAL_EVERY):
    h, s = world["h"], world["s"]
    v = world["c"].copy()
    out = []
    for t, lr in enumerate(lrs, 1):
        v = (1 - lr * h) ** 2 * v + lr * lr * s
        if t % every == 0:
            out.append(world["L_inf"] + 0.5 * np.sum(h * v))
    return np.array(out)


def final_loss(world, lrs):
    return float(run(world, lrs, every=len(lrs))[-1])


# ---------------------------------------------------------------- schedules
def constant(peak, T):
    return np.full(T, peak)


def cosine(peak, T, floor=0.0):
    t = np.arange(T)
    return floor + (peak - floor) * 0.5 * (1 + np.cos(np.pi * t / T))


def wsd(peak, T, frac):
    n_dec = int(T * frac)
    return np.concatenate([np.full(T - n_dec, peak), np.linspace(peak, 0, n_dec + 1)[1:]])


def linear(peak, T):
    return np.linspace(peak, 0, T + 1)[1:]


def step(peak, T):
    lr = np.full(T, peak); lr[T // 2:] = peak / 3; lr[3 * T // 4:] = peak / 10
    return lr


def rewarm(peak, T):
    """cosine to zero over the first half, then re-warm to peak/2 and cosine down again."""
    return np.concatenate([cosine(peak, T // 2), cosine(peak / 2, T - T // 2)])


def cyclic(peak, T, n=4):
    L = T // n
    return np.concatenate([cosine(peak, L) for _ in range(n)])


def warm_cos(peak, T, w=0.1):
    nw = int(T * w)
    return np.concatenate([np.linspace(0, peak, nw + 1)[1:], cosine(peak, T - nw)])


def train_schedules(peak, T=T_TRAIN):
    return {
        "const_hi": constant(peak, T),
        "const_mid": constant(peak / 3, T),
        "const_lo": constant(peak / 10, T),
        "cosine": cosine(peak, T),
    }


VISIBLE_QUERY_TYPES = ["wsd20", "wsd05", "linear", "step", "cos_half_peak"]
HIDDEN_QUERY_TYPES = ["rewarm", "cyclic4", "cos_long", "wsd_long", "warm_cos"]


def query_schedules(peak, T=T_TRAIN, kinds=None):
    q = {
        "wsd20": wsd(peak, T, 0.2),
        "wsd05": wsd(peak, T, 0.05),
        "linear": linear(peak, T),
        "step": step(peak, T),
        "cos_half_peak": cosine(peak / 2, T),
        "rewarm": rewarm(peak, T),
        "cyclic4": cyclic(peak, T),
        "cos_long": cosine(peak, 2 * T),
        "wsd_long": wsd(peak, 2 * T, 0.2),
        "warm_cos": warm_cos(peak, T),
    }
    return {k: v for k, v in q.items() if kinds is None or k in kinds}


# ---------------------------------------------------------------- settings
# (name, seed, gamma, sigma2, peak_lr, T, eval_every, hidden)
SETTINGS = [
    ("dev_a", 11, 0.45, 0.20, 1.0, 2000, 20, False),
    ("dev_b", 12, 0.60, 0.35, 0.5, 2000, 20, False),
    ("dev_c", 13, 0.35, 0.12, 2.0, 2000, 20, False),
    ("hid_a", 101, 0.28, 0.12, 0.7, 1500, 25, True),     # flatter bias decay, low noise
    ("hid_b", 102, 0.75, 0.45, 1.3, 2500, 25, True),     # steep decay, noisy
    ("hid_c", 103, 0.50, 0.25, 0.25, 3000, 30, True),    # small lr scale, long run
    ("hid_d", 104, 0.40, 0.60, 1.0, 1200, 20, True),     # very noisy, short run
]
OBS_NOISE = 0.004


def build_setting(row):
    name, seed, gamma, sigma2, peak, T, every, hidden = row
    w = make_world(seed, gamma, sigma2)
    w["h"] = w["h"] / peak; w["c"] = w["c"] * peak; w["s"] = w["s"] / peak   # lr*h scale invariant
    rng = np.random.default_rng(seed + 7)
    train = {}
    for k, lrs in train_schedules(peak, T).items():
        loss = run(w, lrs, every)
        steps = np.arange(1, len(loss) + 1) * every
        train[k] = dict(lrs=lrs, steps=steps, loss=loss + rng.normal(0, OBS_NOISE, len(loss)))
    kinds = VISIBLE_QUERY_TYPES + (HIDDEN_QUERY_TYPES if hidden else [])
    Q = query_schedules(peak, T, kinds)
    truth = {k: final_loss(w, v) for k, v in Q.items()}
    D = final_loss(w, train_schedules(peak, T)["const_hi"]) - final_loss(w, train_schedules(peak, T)["cosine"])
    return dict(name=name, hidden=hidden, train=train, queries=Q, truth=truth, scale=D)
