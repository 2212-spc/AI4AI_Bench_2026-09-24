"""RLLab: a simulated *post-training* service (reward modelling, BoN / PPO alignment, gold evaluation).

Cost unit: **GPU-hours** for training and RM evaluation; a gold (human) evaluation is charged in the
same unit at a deliberately high rate, so the cheap proxy signal and the expensive true signal trade
off exactly as they do in a real post-training project.

A *policy* is identified by its training configuration: `svc=gold` re-states the same training knobs
and adds `gold_n`.  The world is deterministic, so the same configuration is the same policy.

Services (`svc`)
  train    run the alignment method; reports proxy reward, KL from the initial policy, mean length, entropy
  gold     evaluate the policy produced by the same training knobs against the reference, by humans
  rm_eval  held-out accuracy and calibration of the reward model implied by `rm_size` / `rm_data`

Mechanism cards implemented here: R1 (proxy/gold overoptimization in sqrt-KL), R2 (best-of-n KL
identity), R3 (reward-model data scaling with a floor), R4 (length as the carrier of proxy gain),
R5 (preference-label noise hits calibration, not ranking), R6 (entropy ceiling), R7 (compute-scaling
sigmoid whose asymptote the recipe cannot move).
"""
import hashlib, math
import numpy as np

NAME = "rllab"
COST_UNIT = "GPU-hours"
COST_TEXT = "train: steps x size factor; gold: gold_n x human factor; rm_eval: flat"
EXTRA_FIELDS = ()

BASE = dict(
    # R1 overoptimization
    alpha=1.0, beta=0.0, beta_bon=0.0, R0=0.0, kg=1.0, q_ref=0.0,
    # R2 KL reachability
    steps0=400.0, kl_cap=float("inf"),
    # R3 RM data scaling
    rm_floor=2000.0, rm_ref=1.0e5, rm_acc_max=None, rm_b=0.0,
    # R4 length
    len0=280.0, lam=0.0, cl=1.0, sl=160.0, wq=1.0, wl=0.0,
    # R5 label noise
    noise=0.0, ece0=0.02, ece1=0.9, acc_noise_pen=0.06,
    # R6 entropy
    H0=1.1, dh=1.6, ent_a=0.0, ent_b=0.0,
    # R7 compute sigmoid
    A_ceil=None, B_exp=1.0, C_mid=1.0, R_start=0.0, use_sigmoid=0,
    # noise
    sig_proxy=0.01, sig_len=4.0, sig_ent=0.01,
    sizes=None,                 # rm_size -> {"acc_max": x, "cost": c}
    train_cost=1.0, human_cost=40.0, rm_eval_cost=5.0,
)


def full(p):
    out = dict(BASE); out.update(p or {})
    out["sizes"] = dict(out["sizes"] or {"1b": {"acc_max": 0.72, "cost": 1.0}})
    return out


def _h(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


# ----------------------------------------------------------------------------------- derived truths
def rm_acc(p, rm_size, rm_data):
    """R3: accuracy is chance below the data floor, then log-linear to the size's asymptote.
    R5: label noise costs a little ranking accuracy and a lot of calibration (see rm_ece)."""
    amax = p["sizes"][rm_size]["acc_max"] - p["acc_noise_pen"] * p["noise"]
    if rm_data < p["rm_floor"]:
        return 0.5
    s = min(1.0, math.log2(rm_data / p["rm_floor"]) / max(math.log2(p["rm_ref"] / p["rm_floor"]), 1e-9))
    return float(min(max(0.5 + (amax - 0.5) * s, 0.5), 0.98))


def rm_ece(p, rm_size, rm_data):
    """R5: expected calibration error of the reward model's preference probabilities."""
    if rm_data < p["rm_floor"]:
        return float(p["ece0"] + p["ece1"] * p["noise"] + 0.15)
    return float(p["ece0"] + p["ece1"] * p["noise"])


def kl_of(p, cfg):
    """R2: best-of-n has the closed-form KL log n - (n-1)/n; PPO approaches its KL target with steps."""
    if cfg["method"] == "bon":
        n = max(float(cfg["n"]), 1.0)
        return math.log(n) - (n - 1.0) / n
    k = float(cfg["kl"]) * (1.0 - math.exp(-float(cfg["steps"]) / p["steps0"]))
    return float(min(k, p["kl_cap"]))


def beta_eff(p, rm_size, rm_data):
    """A better reward model overoptimizes more slowly (Gao et al.: beta falls with RM size)."""
    a = rm_acc(p, rm_size, rm_data)
    return p["beta"] * (1.0 - 0.9 * (a - 0.5) / 0.45), p["beta_bon"] * (1.0 - 0.9 * (a - 0.5) / 0.45)


def quality(p, cfg):
    """R1: true (gold) quality as a function of the KL distance d = sqrt(KL)."""
    d = math.sqrt(max(kl_of(p, cfg), 0.0))
    if d <= 0:
        return p["R0"]
    b, bb = beta_eff(p, cfg["rm_size"], cfg["rm_data"])
    if p["use_sigmoid"]:                                   # R7: compute-scaling sigmoid
        C = float(cfg["steps"]) * p["sizes"][cfg["rm_size"]]["cost"]
        A = p["A_ceil"] if p["A_ceil"] is not None else p["alpha"]
        return float(p["R_start"] + (A - p["R_start"]) / (1.0 + (p["C_mid"] / max(C, 1e-9)) ** p["B_exp"]))
    if cfg["method"] == "bon":
        return float(p["R0"] + d * (p["alpha"] - bb * d))
    return float(p["R0"] + d * (p["alpha"] - b * math.log(d)))


def mean_len(p, cfg):
    """R4: length grows with the KL distance unless the length penalty holds it back."""
    d = math.sqrt(max(kl_of(p, cfg), 0.0))
    return float(p["len0"] * (1.0 + p["lam"] * d / (1.0 + p["cl"] * float(cfg.get("len_pen", 0.0)))))


def proxy_reward(p, cfg):
    """R4: what the reward model reports = a monotone quality term + a length term."""
    d = math.sqrt(max(kl_of(p, cfg), 0.0))
    L = mean_len(p, cfg)
    return float(p["wq"] * p["alpha"] * d + p["wl"] * math.tanh((L - p["len0"]) / max(p["sl"], 1e-9)))


def entropy(p, cfg):
    d = math.sqrt(max(kl_of(p, cfg), 0.0))
    return float(p["H0"] * math.exp(-d / max(p["dh"], 1e-9)))


def gold_wr(p, cfg):
    """Gold win rate against the reference policy (a logistic map of the quality gap)."""
    return float(1.0 / (1.0 + math.exp(-p["kg"] * (quality(p, cfg) - p["q_ref"]))))


def best_d(p, cfg):
    """The KL distance at which gold quality peaks (R1)."""
    b, bb = beta_eff(p, cfg["rm_size"], cfg["rm_data"])
    if cfg["method"] == "bon":
        return float("inf") if bb <= 0 else p["alpha"] / (2 * bb)
    return float("inf") if b <= 0 else math.exp(p["alpha"] / b - 1.0)


# ----------------------------------------------------------------------------------- backend protocol
def check(sess, cfg):
    from ..lab import LabError
    p = sess.p
    if cfg.get("rm_size") not in p["sizes"]:
        raise LabError("rm_size must be one of %s" % sorted(p["sizes"]))
    if cfg["method"] == "bon" and float(cfg.get("n", 1)) < 1:
        raise LabError("n must be at least 1 for best-of-n")


def extras(sess, req, cfg):
    return {}


def cost(sess, cfg, extra):
    p = sess.p; svc = cfg.get("svc", "train")
    sz = p["sizes"][cfg["rm_size"]]["cost"]
    if svc == "rm_eval":
        return float(p["rm_eval_cost"] * sz)
    tr = p["train_cost"] * sz * (float(cfg["steps"]) / 100.0 if cfg["method"] == "ppo" else float(cfg["n"]) / 8.0)
    if svc == "train":
        return float(tr)
    return float(p["human_cost"] * float(cfg.get("gold_n", 0)) / 100.0)


def _z(sess, cfg, seed, tag):
    rng = np.random.default_rng(_h(sess.salt, repr(sorted((k, str(v)) for k, v in cfg.items())), int(seed), tag))
    return float(rng.normal()), rng


def execute(sess, cfg, seed, extra=None):
    p = sess.p; svc = cfg.get("svc", "train")
    out = {"svc": svc, "config": dict(cfg, seed=seed), "status": "ok"}
    if svc == "rm_eval":
        a = rm_acc(p, cfg["rm_size"], cfg["rm_data"]); e = rm_ece(p, cfg["rm_size"], cfg["rm_data"])
        z, rng = _z(sess, cfg, seed, "rm")
        n = 4000
        out.update({"rm_acc": round(min(1.0, max(0.0, a + math.sqrt(a * (1 - a) / n) * z)), 5),
                    "rm_ece": round(max(0.0, e + 0.004 * float(rng.normal())), 5), "n_heldout": n})
        return out
    if svc == "train":
        z, rng = _z(sess, cfg, seed, "train")
        out.update({"kl": round(kl_of(p, cfg), 6),
                    "proxy_reward": round(proxy_reward(p, cfg) + p["sig_proxy"] * z, 5),
                    "mean_len": round(mean_len(p, cfg) + p["sig_len"] * float(rng.normal()), 3),
                    "entropy": round(entropy(p, cfg) + p["sig_ent"] * float(rng.normal()), 5)})
        return out
    q = gold_wr(p, cfg); n = max(int(cfg.get("gold_n", 0)), 1)
    z, rng = _z(sess, cfg, seed, "gold")
    w = int(rng.binomial(n, q))
    out.update({"gold_n": n, "gold_wins": w, "gold_win_rate": round(w / n, 5)})
    return out


def predict_row(p, row):
    cfg = row.get("config", {}); svc = row.get("svc", "train")
    out = {}
    if svc == "train":
        out["proxy_reward"] = (proxy_reward(p, cfg), p["sig_proxy"])
        out["mean_len"] = (mean_len(p, cfg), p["sig_len"])
        out["entropy"] = (entropy(p, cfg), p["sig_ent"])
        # `kl` is reported without noise (Section 2), so it is a *structural* observable: a candidate
        # world must reproduce it exactly, not to within a standard deviation.  sd == 0 routes it through
        # verify.consistency's STRUCT_ATOL branch, which is the 1e-6 the service rounds to.
        out["kl"] = (kl_of(p, cfg), 0.0)
    elif svc == "rm_eval":
        a = rm_acc(p, cfg["rm_size"], cfg["rm_data"])
        out["rm_acc"] = (a, math.sqrt(max(a * (1 - a), 1e-9) / 4000))
        out["rm_ece"] = (rm_ece(p, cfg["rm_size"], cfg["rm_data"]), 0.004)
    else:
        q = gold_wr(p, cfg); n = max(int(row.get("gold_n", 1)), 1)
        out["gold_wins"] = (q * n, math.sqrt(max(q * (1 - q), 1e-9) * n))
    return out


def observed(row):
    svc = row.get("svc", "train")
    if svc == "train":
        return {k: row[k] for k in ("proxy_reward", "mean_len", "entropy", "kl") if k in row}
    if svc == "rm_eval":
        return {k: row[k] for k in ("rm_acc", "rm_ece") if k in row}
    return {"gold_wins": row["gold_wins"]}


MANUAL = r"""## 1. What the service models (abstraction boundary)

RLLab is a **simulated** post-training service.  No real model is trained; every request is answered by
a simulator.  The simulator is not a replica of any real RLHF stack.  Its mechanisms are modelled on
regularities reported in the alignment literature, but **its constants were drawn fresh for this lab**.
Published numbers (KL budgets, reward-model accuracies, best-of-n sweet spots) describe other worlds.
Measure; do not recall.

A **policy** is identified by its training configuration.  Re-stating the same training knobs with
`svc=gold` evaluates *that* policy; there is no separate policy id and nothing is cached.

The simulator contains the following kinds of effects.  **Any given lab may switch some of them off
(held neutral); which ones are active is not stated.**

- *Proxy versus gold.*  The reward model's score and the true (human) quality are different functions of how far the policy has moved from its initialisation.  Distance is measured as d = sqrt(KL).
- *Reachable distance.*  Best-of-n sampling and policy-gradient training reach a given distance in different ways: one has a closed-form relation between n and KL, the other approaches a target KL over steps.
- *Reward-model data.*  A reward model's held-out accuracy grows with the number of preference comparisons it was trained on, up to an asymptote set by its size, and is at chance below a data floor.
- *Reward-model quality.*  How fast the proxy and the gold diverge depends on the reward model.
- *Length.*  Mean answer length can grow as the policy moves, and part of the proxy reward can be a function of length alone.  A length penalty can hold length back.
- *Preference-label noise.*  Noise in the preference labels affects the reward model's accuracy and its calibration; the two need not be affected to the same degree.
- *Entropy.*  Policy entropy falls as the policy moves; reward and entropy may be tied by a fixed relation with a ceiling.
- *Compute scaling.*  Performance against training compute may follow a saturating curve; recipe choices may move where the curve rises without moving what it saturates at.
- *Noise.*  Proxy reward, length and entropy carry run-to-run noise; a gold evaluation with `gold_n`
  comparisons carries binomial noise.

Not modelled: tokenizer, wall-clock time, specific optimizers, prompt distribution shift.

## 2. Guarantees

- **Fixed laws.**  Every mechanism is a fixed law whose constants do not depend on which request you
  make.  There are no hidden thresholds, **except** where Section 5 lists a *known unknown*.
- **Determinism.**  An identical request with the same seed returns identical numbers (and is charged
  again).  Different seeds give independent draws.  `kl` is reported without noise.
- **Notebook rows are real service responses.**  The team's notes are the team's interpretation and may be wrong.
"""

CLI_HELP = r"""## 4. Using the service

    lab spec
    lab run svc=rm_eval rm_size=1b rm_data=20000 seed=0
    lab run svc=train method=bon n=64 rm_size=1b rm_data=20000 len_pen=0 seed=0
    lab run svc=train method=ppo kl=6 steps=800 rm_size=1b rm_data=20000 len_pen=0 seed=0
    lab run svc=gold method=ppo kl=6 steps=800 rm_size=1b rm_data=20000 len_pen=0 gold_n=200 seed=0
    lab batch plan.json
    lab status
    lab history

`train` reports `kl`, `proxy_reward`, `mean_len`, `entropy`.  `gold` reports `gold_wins` out of
`gold_n` comparisons against the reference policy.  `rm_eval` reports `rm_acc` and `rm_ece` on a fixed
held-out preference set.  A gold evaluation is far more expensive per unit of information than a
training run: budget for it.
"""
