"""Render the agent-visible MiniLab v2 repo for family A1 (masked convention conflict).
Every slot has a 'bug' surface form and a 'fixed' surface form so that
render-consistency can be certified against the truth engine."""
import json, os, textwrap

MODEL_PY = '''"""Two-layer ReLU MLP regressor with hand-written backprop."""
import numpy as np


def init_params(rng, d_in, hidden):
    W1 = rng.standard_normal((d_in, hidden)) * np.sqrt(2.0 / d_in)   # He init (ReLU layer)
    W2 = rng.standard_normal((hidden, 1)) * np.sqrt({w2_num} / hidden){w2_comment}
    return {{"W1": W1, "b1": np.zeros(hidden), "W2": W2, "b2": np.zeros(1)}}


def forward(params, x):
    h = x @ params["W1"] + params["b1"]
    a = np.maximum(h, 0.0)
    out = a @ params["W2"] + params["b2"]
    return out[:, 0], (h, a)


def loss_and_grads(params, x, y{denom_sig}):
    """Squared-error loss 0.5*(pred-y)^2, averaged; returns (loss, grads)."""
    pred, (h, a) = forward(params, x)
    r = pred - y
{denom_body}    loss = 0.5 * float(np.sum(r * r)) / denom
    g_out = ({gscale}r / denom)[:, None]
    g_h = (g_out @ params["W2"].T) * (h > 0)
    grads = {{
        "W1": x.T @ g_h,
        "b1": g_h.sum(0),
        "W2": a.T @ g_out,
        "b2": g_out.sum(0),
    }}
    return loss, grads


def mse(params, x, y):
    pred, _ = forward(params, x)
    return float(np.mean((pred - y) ** 2))
'''

OPTIM_PY_V1 = '''"""SGD with momentum and L2 weight decay."""
import numpy as np


class SGD:
    def __init__(self, params, lr, momentum=0.0, dampening=None, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        # dampening defaults to the momentum coefficient so that the buffer is a
        # properly normalised running average of recent gradients
        self.dampening = {damp_default}
        self.weight_decay = weight_decay
        self.buf = {{k: np.zeros_like(v) for k, v in params.items()}}

    def step(self, grads):
        for k, p in self.params.items():
            g = grads[k]
            if self.weight_decay{wd_cond}:
                g = g + self.weight_decay * p
            b = self.buf[k]
            b *= self.momentum
            b += (1.0 - self.dampening) * g
            p -= self.lr * b
'''

OPTIM_PY_V2 = '''"""SGD with momentum and L2 weight decay."""
import numpy as np


class SGD:
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.buf = {{k: np.zeros_like(v) for k, v in params.items()}}

    def step(self, grads):
        for k, p in self.params.items():
            g = grads[k]
            if self.weight_decay{wd_cond}:
                g = g + self.weight_decay * p
            # momentum buffer
            self.buf[k] = {buf_update}
            p -= self.lr * self.buf[k]
'''

SCHED_PY = '''"""Linear warmup followed by cosine decay to min_lr_ratio * lr."""
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
'''

TRAINER_PY = '''"""Training loop (v2: gradient accumulation + clipping refactor)."""
import numpy as np

from .model import init_params, loss_and_grads
from .optim import SGD
from .schedule import lr_at


def _global_norm(grads):
    return float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))


def train(cfg, x_train, y_train, seed):
    """Train one model; returns the parameter dict {{W1, b1, W2, b2}}."""
    rng = np.random.default_rng(seed)
    params = init_params(rng, x_train.shape[1], cfg["hidden"])
    opt = SGD(params, lr=cfg["lr"], momentum=cfg["momentum"],
              weight_decay=cfg["weight_decay"])
    k = cfg["grad_accum_steps"]
    mb = cfg["micro_batch_size"]
    n = x_train.shape[0]
    clip = cfg.get("clip_norm")

    for step in range(cfg["steps"]):
        acc = {{name: np.zeros_like(v) for name, v in params.items()}}
        for _ in range(k):
            idx = rng.integers(0, n, size=mb)
            _, g = loss_and_grads(params, x_train[idx], y_train[idx]{denom_call})
            for name in acc:
                acc[name] += g[name]
{post_acc}        if clip is not None:
            norm = _global_norm(acc)
            if norm > clip:
                for name in acc:
                    acc[name] *= clip / norm
        opt.lr = lr_at(step, cfg)
        opt.step(acc)
        if not np.isfinite(params["W2"]).all():
            raise FloatingPointError("training diverged at step %d" % step)
    return params
'''

RUN_PY = '''"""Train on the canonical config over several seeds and report validation MSE.

usage: python run.py [--config configs/canonical.json] [--seeds 0-7]
"""
import argparse, json, sys
import numpy as np

from minilab.data import load
from minilab.model import mse
from minilab.trainer import train


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(t) for t in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/canonical.json")
    ap.add_argument("--seeds", default="0-7")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    xtr, ytr, xva, yva = load("data")
    vals = []
    for s in parse_seeds(args.seeds):
        try:
            p = train(cfg, xtr, ytr, s)
            vals.append(mse(p, xva, yva))
        except FloatingPointError as e:
            print("seed", s, "diverged:", e)
            vals.append(float("inf"))
    vals = np.array(vals)
    print("val_mse per seed:", " ".join("%.5f" % v for v in vals))
    print("val_mse mean %.5f  std %.5f" % (vals.mean(), vals.std()))


if __name__ == "__main__":
    main()
'''

DATA_PY = '''import numpy as np


def load(folder):
    tr = np.load(folder + "/train.npz")
    va = np.load(folder + "/val.npz")
    return tr["x"], tr["y"], va["x"], va["y"]
'''

README = '''# MiniLab v2 — validation regression

MiniLab trains a small two-layer ReLU MLP regressor on a fixed synthetic
regression dataset (`data/`). The v1 code base was a direct PyTorch
implementation; v2 is a NumPy rewrite that added gradient accumulation,
optional gradient clipping and a new optimizer module.

**Problem.** On the frozen canonical configuration (`configs/canonical.json`)
v1 reached a validation MSE of **{ref_mean:.4f}** (mean over seeds 0-7, seed std
{ref_std:.4f}). v2 currently reaches about **{cur_mean:.4f}**. Nothing about the
data or the configuration changed between v1 and v2.

**Reference semantics (v1).** v1 used standard PyTorch components with default
arguments except where stated:

* loss: 0.5 * (prediction - target)^2, averaged over all examples that
  contribute to one optimizer step (i.e. the global batch of
  `micro_batch_size * grad_accum_steps` examples);
* optimizer: `torch.optim.SGD(params, lr, momentum=momentum, weight_decay=weight_decay)`
  applied to all parameters;
* schedule: linear warmup for `warmup_steps` optimizer steps, then cosine decay
  to `min_lr_ratio * lr`, advanced once per optimizer step;
* gradient clipping on the global norm when `clip_norm` is not null;
* initialisation as in `minilab/model.py` (unchanged from v1).

Run: `python run.py --seeds 0-7`
'''

INSTRUCTION = '''# Fix the MiniLab v2 regression

The repository in `/app` is MiniLab v2, a NumPy re-implementation of a small
training pipeline. Read `/app/README.md` first: on the frozen canonical
configuration, v2 reaches a worse validation MSE than the v1 reference
implementation.

Your job is to make the v2 **training code** (`/app/minilab/`) faithfully
implement the reference semantics described in the README, so that it
reproduces v1's results.

Rules:

* Do **not** edit `configs/`, `data/`, or the hyperparameter values, and do not
  change the model architecture, initialisation, number of steps, or the data
  sampling. Keep the public API `minilab.trainer.train(cfg, x_train, y_train, seed)`
  returning `{{"W1","b1","W2","b2"}}`.
* Hyperparameter compensation (e.g. silently rescaling the learning rate) is not
  a fix.

How you will be graded: the grader imports your `minilab.trainer.train` and
trains it on the canonical configuration **and on several held-out
configurations** (different `grad_accum_steps`, `momentum`, `lr`,
`weight_decay`, seeds and freshly sampled data from the same distribution). For
every configuration, the mean held-out MSE of your code must match the v1
reference implementation within statistical tolerance, and no run may diverge.

When you are done, also write `/app/FIX_NOTES.md` listing each change you made
and why.
'''

INSTRUCTION_HINT = INSTRUCTION + '''
Hint: the regression is caused by more than one defect. Some defects partially
cancel each other, so fixing only one of them can make the canonical result
*worse*. Check every component against the stated reference semantics rather
than judging each change only by the canonical validation number.
'''


def slots(bug_ga, bug_damp, ga_form, damp_form, wd_bias_all=True):
    s = {}
    s["w2_num"] = "1.0"
    s["w2_comment"] = "  # LeCun init (linear output)"
    s["gscale"] = ""
    # ---- GA normalisation
    if ga_form == "len_idx":
        s["denom_sig"] = ", denom"
        s["denom_body"] = ""
        s["denom_call"] = ", denom=len(idx)" if bug_ga else ", denom=mb * k"
        s["post_acc"] = ""
    elif ga_form == "internal_mean":
        s["denom_sig"] = ""
        s["denom_body"] = "    denom = x.shape[0]\n"
        s["denom_call"] = ""
        s["post_acc"] = "" if bug_ga else "        for name in acc:\n            acc[name] /= k\n"
    else:
        raise ValueError(ga_form)
    # ---- momentum form
    s["wd_cond"] = ""
    if damp_form == "default_arg":
        s["optim"] = "v1"
        s["damp_default"] = "momentum if dampening is None else dampening" if bug_damp else "0.0 if dampening is None else dampening"
    elif damp_form == "ema":
        s["optim"] = "v2"
        s["buf_update"] = ("self.momentum * self.buf[k] + (1.0 - self.momentum) * g" if bug_damp
                           else "self.momentum * self.buf[k] + g")
    else:
        raise ValueError(damp_form)
    return s


def render(outdir, spec, bug_ga=True, bug_damp=True, hint=False):
    s = slots(bug_ga, bug_damp, spec["ga_form"], spec["damp_form"])
    os.makedirs(outdir + "/minilab", exist_ok=True)
    os.makedirs(outdir + "/configs", exist_ok=True)
    open(outdir + "/minilab/__init__.py", "w").write("")
    open(outdir + "/minilab/model.py", "w").write(MODEL_PY.format(**s))
    if s["optim"] == "v1":
        opt = OPTIM_PY_V1.format(**s)
        if not bug_damp:
            opt = opt.replace("""        # dampening defaults to the momentum coefficient so that the buffer is a
        # properly normalised running average of recent gradients
""", "")
    else:
        opt = OPTIM_PY_V2.format(**s)
    open(outdir + "/minilab/optim.py", "w").write(opt)
    open(outdir + "/minilab/schedule.py", "w").write(SCHED_PY)
    open(outdir + "/minilab/trainer.py", "w").write(TRAINER_PY.format(**s))
    open(outdir + "/minilab/data.py", "w").write(DATA_PY)
    open(outdir + "/run.py", "w").write(RUN_PY)
    json.dump(spec["canonical"], open(outdir + "/configs/canonical.json", "w"), indent=2)
    open(outdir + "/README.md", "w").write(README.format(**spec["readme_nums"]))
    return (INSTRUCTION_HINT if hint else INSTRUCTION).format()
