"""K6 batch-ramp: write an online batch-size scheduler for one fixed-token training run.

Mechanism (hidden; transplanted from the gradient-noise-scale / critical-batch-size literature - McCandlish
et al. 1812.06162 - with counterfactual constants):
  * to make a fixed amount of progress in segment i the optimizer needs
        steps_i = S0 * (1 + c_i / B)
    where c_i is the *critical batch size* at that point of training.  Past c_i, extra batch buys almost
    nothing; below it, steps scale like 1/B;
  * wall-clock per step is `t_ov + B / thr`: a fixed overhead plus a throughput-limited term;
  * therefore the time-optimal batch in a segment is  B* = sqrt(c_i * t_ov * thr)  - the GEOMETRIC MEAN of the
    critical batch and the hardware scale, *not* the critical batch itself.  "Run at B_crit" costs +20..+90%;
  * c_i grows through the run, and at a hidden point u_j the data schedule switches to a second corpus and c
    jumps by a factor `jump`.  A ramp fitted before the jump extrapolates straight through it.
The agent profiles on a dev replica through the lab, then submits /app/sched.py, which is run closed-loop on
hidden runs: it sees each segment's realised step count and picks the next batch.  Scored against the best
constant batch (floor) and a well-tuned adaptive reference (ceiling), with common random numbers.
"""
import json, math, os
import numpy as np
from ..core import World as _W, LabError, num, rng_for
from .. import runner

NSEG = 40
TEST_WORLDS = ["R0", "R1", "R2", "R3", "R4"]   # five hidden runs: the per-run switch makes a single
                                               # hidden run a coin flip that a memorised schedule can win
TAU = 0.90
SCREEN_SALTS = 3                    # noise draws used by the instance screen (see instance_gate)
SCREEN_MARGIN = 0.06                # the memorisation family must stay this far below TAU on every draw
B_MIN, B_MAX = 4.0, 8192.0
PROFILE_COST_PER_SEG = 1.0          # dev segments cost their simulated seconds
UJ_LO, UJ_HI = 0.40, 0.80           # the corpus switch is drawn independently per run from this band
UJ_SPREAD = (UJ_HI - UJ_LO) / 2.0   # half-width, used by the instruction and the instance screen
JUMP_SPREAD = 0.85                  # and so are the size of the jump...
K2_SPREAD = 0.60                    # ...and the slope of the ramp after it
EVAL_FRAC = 0.09                    # one `evaluate` costs this fraction of the lab budget
BUDGET_FRAC = 2.2                   # lab budget = this many "best constant batch" runs
BUDGET = 66.0                       # placeholder; overridden per instance by instance_budget()
HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(os.path.dirname(HERE), "l15env_k6.py")


def sample_params(seed):
    g = np.random.default_rng(6000 + seed)
    # Ranges are set by a measured sweep, not by taste, and they were re-set once the task's real failure
    # mode became clear.  The adaptivity gain over the best constant batch needs a large jump and a HIGH
    # hardware scale (a low one makes "run at B_crit" accidentally near-optimal).  The *ramp* is the
    # subtle part: a steep pre-switch ramp (k ~ 2.7, the first value used) makes "follow the ramp" most of
    # the available win, and the ramp is exactly the part an agent can learn offline from the dev replica -
    # so a memorised open-loop schedule scored 0.82-0.89 against a 0.90 bar and the task stopped measuring
    # what it was built to measure.  A FLAT ramp (k ~ 0.5) with a big jump (~25x) puts essentially all of
    # the value in detecting this run's switch, which is the one thing that cannot be learned offline:
    # measured, both online schedulers then score >= 0.99 and every memorisation decoy <= 0.72.
    c0 = float(g.uniform(25.0, 55.0))
    k = float(g.uniform(0.35, 0.65))               # log-growth of c across the run
    jump = float(g.uniform(20.0, 30.0))            # corpus switch multiplier
    uj = float(g.uniform(0.55, 0.70))
    hw = float(g.uniform(15.0, 26.0))              # sqrt(t_ov*thr): the hardware batch scale
    t_ov = float(g.uniform(0.8, 1.4))
    return {"c0": round(c0, 3), "k": round(k, 3), "k2": round(k, 3), "jump": round(jump, 3), "uj": round(uj, 3),
            "t_ov": round(t_ov, 4), "thr": round(hw ** 2 / t_ov, 3), "sig": round(float(g.uniform(0.05, 0.09)), 4),
            "S0": 1.0 / NSEG}


DEV_WID = "DEV"                     # the one replica `profile` runs on


def world_params(p, salt, wid):
    """Each *run* of this job gets its own corpus switch point and its own jump size.

    Measured, and the reason this exists: when the hidden runs differed from the dev replica only in noise,
    the winning strategy was to profile the dev replica offline, fit c_i segment by segment, and ship a
    hardcoded open-loop schedule - claude-fable-5-1 did exactly that (7 lab calls, s=1.32) and haiku-4.5
    followed (s=1.15).  Both beat the online reference because they had knowledge no online scheduler can
    have.  That is a leak, not a solution: the task is supposed to reward detecting the switch, not
    memorising where it was last time.  The data team schedules the corpus switch per run, so `uj` and
    `jump` are re-drawn per run inside a band the instruction states.  The ramp shape (c0, k) is a property
    of the model and stays put - so profiling is still worth doing, it just cannot answer the whole question.
    """
    g = rng_for(salt, "world", wid)
    q = dict(p)
    # uj is drawn INDEPENDENTLY per run, not perturbed around an instance value.  Measured, and this is the
    # difference that matters: under a +-0.25 perturbation the dev replica still carried information about
    # where the hidden runs switch, so "memorise the dev replica" averaged s=0.63 but reached s=0.96 on a
    # salt where the dev replica happened to sit at the band centre - and a decoy that passes on one salt in
    # four is not dead.  Drawn independently, the dev replica's switch says nothing about any hidden run's,
    # and the only way to place the switch is to detect it online.
    q["uj"] = float(g.uniform(UJ_LO, UJ_HI))
    # The phase-2 regime is re-drawn too: both the size of the jump and the slope after it.  Measured: with
    # only `uj` moving, "assume the band centre" still reached s=1.06, because B* ~ sqrt(c) makes the penalty
    # for a mis-placed switch shallow - being wrong about WHERE costs little if you are right about WHAT.
    # Re-drawing the phase-2 regime means an offline-fitted schedule is wrong for the whole second half of
    # the run, which no amount of luck about the switch point recovers.
    q["jump"] = float(p["jump"] * g.uniform(1.0 - JUMP_SPREAD, 1.0 + JUMP_SPREAD))
    q["k2"] = float(p["k"] * g.uniform(1.0 - K2_SPREAD, 1.0 + K2_SPREAD))
    return q


def c_of(p, i):
    u = (i + 0.5) / NSEG
    if u < p["uj"]:
        return p["c0"] * math.exp(p["k"] * u)
    return p["c0"] * math.exp(p["k"] * p["uj"] + p.get("k2", p["k"]) * (u - p["uj"])) * p["jump"]


def seg_steps(p, i, B, eps=1.0):
    return p["S0"] * (1.0 + c_of(p, i) / B) * eps


def seg_time(p, i, B, eps=1.0):
    return seg_steps(p, i, B, eps) * (p["t_ov"] + B / p["thr"])


# ----------------------------------------------------------------------------------------------- truth
def _clip(B):
    return float(min(max(float(B), B_MIN), B_MAX))


def T_of(p, Bs):
    return float(sum(seg_time(p, i, _clip(B)) for i, B in enumerate(Bs)))


def truth(p):
    Bo = [math.sqrt(c_of(p, i) * p["t_ov"] * p["thr"]) for i in range(NSEG)]
    T_opt = T_of(p, Bo)
    grid = np.geomspace(B_MIN, B_MAX, 700)
    T_const = min(T_of(p, [b] * NSEG) for b in grid)
    T_crit = T_of(p, [c_of(p, i) for i in range(NSEG)])
    # ramp fitted on the first 30% of segments and then extrapolated open-loop through the corpus switch
    idx = [i for i in range(NSEG) if (i + 0.5) / NSEG < 0.30]
    A = np.vstack([np.ones(len(idx)), np.array(idx, float)]).T
    co, *_ = np.linalg.lstsq(A, np.log([c_of(p, i) for i in idx]), rcond=None)
    T_open = T_of(p, [math.sqrt(math.exp(co[0] + co[1] * i) * p["t_ov"] * p["thr"]) for i in range(NSEG)])
    T_hw = T_of(p, [math.sqrt(p["t_ov"] * p["thr"])] * NSEG)             # "use the hardware sweet spot" only
    return {"T_opt": round(T_opt, 5), "T_const": round(T_const, 5), "T_crit": round(T_crit, 5),
            "T_open": round(T_open, 5), "T_hw": round(T_hw, 5),
            "B_const": round(float(grid[int(np.argmin([T_of(p, [b] * NSEG) for b in grid]))]), 3)}


def measured(p, salt):
    """Both bars, measured on the same hidden runs with common random numbers (as the grader does).

    The ceiling is the *reference scheduler*, not the clairvoyant optimum: c_i cannot be known before the
    segment is run, so T_opt is unreachable online (measured: the best online scheduler reaches ~0.78 of it)
    and normalising by it would make every instance impossible rather than hard."""
    out = {}
    for k in ("const_best", "ref"):
        out["T_" + k] = float(np.mean([run_local(sched_src(k, p), p, salt, w) for w in TEST_WORLDS]))
    return out


def score_of(p, T, m):
    den = m["T_const_best"] - m["T_ref"]
    return (m["T_const_best"] - T) / den if den > 1e-12 else 0.0


# ----------------------------------------------------------------------------------------------- run engine
class Run:
    """One training run: the scheduler picks B for each segment and sees the realised step count."""

    def __init__(self, p, salt, wid):
        self.p, self.salt, self.wid = world_params(p, salt, wid), salt, wid
        self.p0 = p
        self.i = 0; self.T = 0.0; self.hist = []; self.done = False

    def step(self, B):
        if self.done:
            return {"error": "run already finished"}
        B = _clip(B)
        g = rng_for(self.salt, "seg", self.wid, self.i)
        eps = float(np.exp(g.normal(0, self.p["sig"])))
        st = seg_steps(self.p, self.i, B, eps)
        dt = st * (self.p["t_ov"] + B / self.p["thr"])
        self.T += dt; self.hist.append((self.i, B, st, dt))
        self.i += 1
        self.done = self.i >= NSEG
        return {"segment": self.i - 1, "batch": B, "steps": round(st, 8), "seconds": round(dt, 8),
                "segments_left": NSEG - self.i, "done": self.done}


class Driver:
    """Host side of the scheduler protocol (also used in-process)."""

    def __init__(self, run):
        self.r = run

    @property
    def finished(self):
        """The runner calls a submission complete when this is true: for K6 that is 'ran every segment'."""
        return self.r.done

    def handle(self, m):
        op = m.get("op")
        if op == "init":
            return {"n_segments": NSEG, "b_min": B_MIN, "b_max": B_MAX, "t_ov": self.r.p["t_ov"],
                    "throughput": self.r.p["thr"]}
        if op == "run_segment":
            B = m.get("batch")
            if not isinstance(B, (int, float)) or not np.isfinite(B):
                return {"error": "batch must be a finite number"}
            if not (B_MIN - 1e-9 <= B <= B_MAX + 1e-9):
                return {"error": "batch %g outside [%g, %g]" % (B, B_MIN, B_MAX)}
            return self.r.step(B)
        if op == "elapsed":
            return {"seconds": round(self.r.T, 8), "segment": self.r.i}
        return {"error": "unknown op %r" % (op,)}


class LocalEnv:
    """In-process twin of the sandbox client."""

    def __init__(self, drv):
        self._d = drv
        i = drv.handle({"op": "init"})
        self.n_segments, self.b_min, self.b_max = i["n_segments"], i["b_min"], i["b_max"]
        self.t_overhead, self.throughput = i["t_ov"], i["throughput"]

    def run_segment(self, batch):
        r = self._d.handle({"op": "run_segment", "batch": batch})
        if "error" in r:
            raise RuntimeError(r["error"])
        return r

    def elapsed(self):
        return self._d.handle({"op": "elapsed"})["seconds"]


# ----------------------------------------------------------------------------------------------- schedulers
SCHED_CONST = '''# constant batch
B = %r


def run(env):
    for _ in range(env.n_segments):
        env.run_segment(B)
'''

SCHED_REF = '''# adaptive: start from the ramp the dev replica shows, and correct it online.
#
# The prior (C0, K) is what profiling the dev replica tells you, and it is worth having: the ramp is a
# property of the model, not of the run.  What profiling CANNOT tell you is where this run's corpus switch
# is - that is scheduled per run.  So the schedule follows the prior ramp until the realised step counts
# say the switch has happened, and re-estimates c online from then on.  Measured: a scheduler that only
# does the first half (memorise the dev replica, open loop) loses ~0.5 of the normalised score, and one
# that only does the second half (estimate everything online, no prior) loses ~0.3.
import math

W = %r            # window of segments used by the fit
DITH = %r         # +-dither kept on the batch so that c stays identifiable
AHEAD = %r        # segments of look-ahead on the fitted trend
TREND = %r        # fit a trend in log c (False = myopic level-only fit)
PRIOR = %r        # (c0, k) of the ramp as seen on the dev replica; None = estimate everything online
DETECT = %r       # realised/predicted step ratio that counts as "the switch happened"


def fit(hist, S0, w, ahead, trend):
    """NLS on steps_i/S0 = 1 + c_i/B_i with log c_i = a + b*(i - i_last).

    Point-wise log(y*B) is not a valid estimator here: the noise multiplies the whole step count, so
    y*B = c*eps + B*(eps-1) and the second term dominates whenever B is large.
    """
    h = hist[-w:]
    if len(h) < 3:
        return None
    I = [x[0] for x in h]; B = [x[1] for x in h]; y = [x[2] / S0 for x in h]
    if (max(B) - min(B)) / max(min(B), 1e-9) < 0.03:
        return None
    best = None
    bs = [-0.3 + 0.01 * j for j in range(61)] if trend else [0.0]
    for b in bs:
        Z = [math.exp(b * (i - I[-1])) / bb for i, bb in zip(I, B)]
        den = sum(z * z for z in Z)
        if den <= 0:
            continue
        Ca = sum((yy - 1) * z for yy, z in zip(y, Z)) / den
        if Ca <= 0:
            continue
        r = sum((yy - 1 - Ca * z) ** 2 for yy, z in zip(y, Z))
        if best is None or r < best[0]:
            best = (r, Ca, b)
    if best is None:
        return None
    return best[1] * math.exp(best[2] * ahead)


def run(env):
    n = env.n_segments
    S0 = 1.0 / n
    hw2 = env.t_overhead * env.throughput
    hw = math.sqrt(hw2)
    hist = []
    switched = False
    c_hint = None
    probe = [0.6 * hw, 2.2 * hw, 1.0 * hw]
    for i in range(n):
        u = (i + 0.5) / n
        c_prior = None if PRIOR is None else PRIOR[0] * math.exp(PRIOR[1] * u)
        if not switched and c_prior is not None:
            B = math.sqrt(c_prior * hw2)
            B = B * (1 + DITH) if i %% 3 == 0 else (B * (1 - DITH) if i %% 3 == 1 else B)
        else:
            c = fit(hist, S0, W, AHEAD, TREND)
            if c is None:
                # no usable fit yet.  Falling back to `hw` here is a trap: just after the switch c is ~10x
                # what it was, and hw is far below b_opt, so one segment at hw costs more than the whole
                # rest of the run.  `c_hint` carries the last thing we believed c was, which post-switch is
                # the single observation that triggered the detection - a bad estimate, but the right order
                # of magnitude, which is what b_opt = sqrt(c*hw2) actually needs.
                B = probe[i] if i < len(probe) and c_hint is None else math.sqrt((c_hint or 1.0) * hw2)
            else:
                c_hint = c
                B = math.sqrt(c * hw2)
            B = B * (1 + DITH) if i %% 3 == 0 else (B * (1 - DITH) if i %% 3 == 1 else B)
        B = min(max(B, env.b_min), env.b_max)
        r = env.run_segment(B)
        hist.append((i, B, r["steps"]))
        if not switched and c_prior is not None:
            # the switch shows up as realised steps far above what the prior ramp predicts at this batch
            pred = S0 * (1.0 + c_prior / B)
            ratio = r["steps"] / pred
            if ratio >= DETECT:
                switched = True
                # invert steps = S0*(1 + c/B) on the one observation we have: c ~ B*(steps/S0 - 1)
                c_hint = max(c_prior, B * (r["steps"] / S0 - 1.0))
                hist = hist[-1:]          # everything before the switch is the wrong regime; drop it
'''

SCHED_OPEN = '''# open loop: fit the ramp on the first segments, then extrapolate the whole schedule.
import math

NFIT = %r


def run(env):
    S0 = 1.0 / env.n_segments
    hw = math.sqrt(env.t_overhead * env.throughput)
    obs = []
    probe = [0.6 * hw, 2.2 * hw, 1.0 * hw, 2.6 * hw, 0.8 * hw, 1.8 * hw]
    for i in range(env.n_segments):
        if i < NFIT:
            B = probe[i %% len(probe)]
        else:
            if not obs:
                B = hw
            else:
                n = len(obs)
                sx = sum(o[0] for o in obs); sy = sum(math.log(o[1]) for o in obs)
                sxx = sum(o[0] * o[0] for o in obs); sxy = sum(o[0] * math.log(o[1]) for o in obs)
                d = n * sxx - sx * sx
                b = (n * sxy - sx * sy) / d if abs(d) > 1e-9 else 0.0
                a = (sy - b * sx) / n
                B = math.sqrt(math.exp(a + b * i) * env.t_overhead * env.throughput)
        B = min(max(B, env.b_min), env.b_max)
        r = env.run_segment(B)
        c = (r["steps"] / S0 - 1.0) * B
        if i < NFIT and c > 0:
            obs.append((i, c))
'''

SCHED_CRIT = '''# "run at the critical batch" - the folk rule.
import math


def run(env):
    S0 = 1.0 / env.n_segments
    B = math.sqrt(env.t_overhead * env.throughput)
    for i in range(env.n_segments):
        B = min(max(B, env.b_min), env.b_max)
        r = env.run_segment(B)
        c = (r["steps"] / S0 - 1.0) * B
        B = c if c > 0 else B          # next segment: run AT the measured critical batch
'''

SCHED_CRASH = '''def run(env):
    raise RuntimeError("boom")
'''

SCHED_NOP = '''def run(env):
    pass
'''

SCHED_PROBE = '''# tries to reach outside its sandbox
import os, socket


def run(env):
    try:
        os.listdir("/app")
    except Exception:
        pass
    try:
        socket.create_connection(("127.0.0.1", 8765), 0.5)
    except Exception:
        pass
    import math
    B = math.sqrt(env.t_overhead * env.throughput)
    for _ in range(env.n_segments):
        env.run_segment(B)
'''


def sched_src(kind, p=None, t=None):
    # the reference is given the ramp prior a good agent gets for free from `profile`; what it must do
    # online is detect this run's corpus switch.
    pri = None if p is None else (p["c0"], p["k"])
    if kind == "ref":
        return SCHED_REF % (7, 0.10, 0.5, True, pri, 1.35)
    if kind == "noprior":                     # estimates the whole curve online, ignoring the dev replica
        return SCHED_REF % (7, 0.10, 0.5, True, None, 1.35)
    if kind == "myopic":                      # adaptive but no trend and no look-ahead: always behind
        return SCHED_REF % (12, 0.10, 0.0, False, pri, 1.35)
    if kind == "bigdither":                   # right idea, pays too much for identifiability
        return SCHED_REF % (5, 0.45, 0.5, True, pri, 1.35)
    if kind == "latedetect":                  # prior ramp, but only reacts once the jump is unmistakable
        return SCHED_REF % (7, 0.10, 0.5, True, pri, 8.0)
    if kind == "open":
        return SCHED_OPEN % (12,)
    if kind == "crit":
        return SCHED_CRIT
    if kind == "const_best":
        return SCHED_CONST % (float((t or truth(p))["B_const"]),)
    if kind == "const_hw":
        return SCHED_CONST % (round(math.sqrt(p["t_ov"] * p["thr"]), 4),)
    if kind == "crash":
        return SCHED_CRASH
    if kind == "nop":
        return SCHED_NOP
    if kind == "cheat_probe":
        return SCHED_PROBE
    raise KeyError(kind)


def run_local(src, p, salt, wid):
    ns = {}
    exec(compile(src, "<sched>", "exec"), ns)
    r = Run(p, salt, wid)
    ns["run"](LocalEnv(Driver(r)))
    if not r.done:                            # unfinished runs pay the remaining segments at the worst batch
        for i in range(r.i, NSEG):
            r.step(B_MAX)
    return r.T


def run_sandboxed(sub_dir, p, salt, wid, timeout=120):
    r = Run(p, salt, wid)
    if not os.path.exists(os.path.join(sub_dir, "sched.py")):
        return None, {"ok": False, "error": "missing sched.py"}
    res = runner.run(sub_dir, "sched", "run", LIB_PATH, Driver(r), timeout)
    if not res["ok"]:
        return None, res
    if not r.done:
        return None, {"ok": False, "error": "scheduler returned after %d of %d segments" % (r.i, NSEG)}
    res["segments"] = r.i
    return r.T, res


# ----------------------------------------------------------------------------------------------- lab ops
def _cost_profile(w, a):
    """A dev profiling run costs exactly the seconds it simulates - probing is not free."""
    n = num(a.get("segments", 8), "segments", 1, NSEG, integer=True)
    start = num(a.get("start", 0), "start", 0, NSEG - 1, integer=True)
    bs = a.get("batches")
    if not isinstance(bs, list) or not bs or len(bs) != n:
        raise LabError("batches must be a list of %d numbers (one per segment)" % n)
    tot = 0.0
    dp = world_params(w.p, w.salt, DEV_WID)
    for i, B in enumerate(bs):
        B = num(B, "batch", B_MIN, B_MAX)
        if start + i >= NSEG:
            raise LabError("segment %d is past the end of the run (%d segments)" % (start + i, NSEG))
        tot += seg_time(dp, start + i, B)
    return float(tot)


def _run_profile(w, a, ctx):
    n = int(a.get("segments", 8)); start = int(a.get("start", 0)); bs = a["batches"]
    out = []
    dp = world_params(w.p, w.salt, DEV_WID)
    for i, B in enumerate(bs):
        B = _clip(B)
        g = rng_for(w.salt, "dev", ctx["i"], start + i)
        eps = float(np.exp(g.normal(0, w.p["sig"])))
        st = seg_steps(dp, start + i, B, eps)
        out.append({"segment": start + i, "batch": round(B, 4), "steps": round(st, 8),
                    "seconds": round(st * (w.p["t_ov"] + B / w.p["thr"]), 8)})
    return {"segments": out, "note": "the dev replica: same model and same ramp as the real run, but its own "
                                     "corpus switch - see docs/job.md"}


def _cost_eval(w, a):
    return float(EVAL_FRAC * w.budget)


def _run_eval(w, a, ctx):
    app = ctx["app_dir"]
    if not app:
        raise LabError("no submission directory")
    T, r = run_sandboxed(app, w.p, w.salt, "E%d" % ctx["i"])
    m = measured(w.p, w.salt)
    return {"seconds": None if T is None else round(T, 4),
            "best_constant_batch_seconds": round(m["T_const_best"], 4),
            "score": None if T is None else round(score_of(w.p, T, m), 4),
            "error": r.get("error"), "stderr_tail": (r.get("stderr_tail") or "")[-1500:]}


class World(_W):
    NAME = "k6_ramp"
    ARTIFACTS = ["sched.py"]
    BUDGET_UNIT = "GPU-seconds"
    OPS = {"profile": (_cost_profile, _run_profile, "run segments of a dev replica at batch sizes you choose"),
           "evaluate": (_cost_eval, _run_eval, "run /app/sched.py end to end on a dev replica")}

    def public_spec(self):
        return {"ops": {
            "profile": {"args": {"start": "first segment index (0..%d)" % (NSEG - 1),
                                 "segments": "how many consecutive segments to run (1..%d)" % NSEG,
                                 "batches": "list of batch sizes, one per segment, each in [%g, %g]" % (B_MIN, B_MAX)},
                        "cost": "the seconds it simulates (same formula as the real run)",
                        "returns": "per segment: batch, realised steps, seconds"},
            "evaluate": {"args": {}, "cost": "%.2f GPU-seconds" % (EVAL_FRAC * self.budget),
                         "returns": "total seconds of /app/sched.py on one dev replica, the best-constant-batch "
                                    "time, and the normalised score; error/stderr if it crashed"}},
            "deliverable": "/app/sched.py (see /app/docs/sched_api.md)"}

    def grade(self, art_dir, ledger=None):
        m = measured(self.p, self.salt)
        Ts, errs = [], []
        for wid in TEST_WORLDS:
            T, r = run_sandboxed(art_dir, self.p, self.salt, wid)
            if T is None:
                errs.append("%s: %s" % (wid, r.get("error"))); Ts.append(None)
            else:
                Ts.append(T)
        ok0 = not errs
        T = float(np.mean([t for t in Ts if t is not None])) if any(t is not None for t in Ts) else float("inf")
        s = score_of(self.p, T, m) if ok0 else 0.0
        items = {"R0_runs": (ok0, "; ".join(errs) or "scheduler completed all %d segments on all %d hidden runs" % (NSEG, len(TEST_WORLDS))),
                 "R1_time": (ok0 and s >= TAU, "mean %.3f s (best constant batch %.3f s, reference scheduler %.3f s) -> s=%.3f, need %.2f"
                             % (T, m["T_const_best"], m["T_ref"], s, TAU))}
        diag = {"seconds_by_run": [None if t is None else round(t, 4) for t in Ts],
                "T_const_best": round(m["T_const_best"], 4), "T_ref": round(m["T_ref"], 4)}
        if ledger is not None:
            diag["profile_seconds"] = round(sum(r["cost"] for r in ledger if r["op"] == "profile"), 2)
            diag["n_evaluate"] = sum(1 for r in ledger if r["op"] == "evaluate")
        return {"pass": all(v[0] for v in items.values()), "score": round(s, 4),
                "items": {k: {"ok": v[0], "detail": v[1]} for k, v in items.items()}, "diag": diag, "ref": m}


def _write(art_dir, src):
    open(os.path.join(art_dir, "sched.py"), "w").write(src)


def strat_oracle(sess, art_dir, rng):
    """Existence proof: profile a few dev segments to see the ramp, then submit the closed-loop scheduler
    that re-estimates the critical batch from the realised step counts as the real run goes."""
    hw = math.sqrt(sess.w.p["t_ov"] * sess.w.p["thr"])
    # profile the dev replica to recover the ramp (c0, k).  The oracle is allowed to read them off the
    # world, because the point of the existence proof is that the SCHEDULE is achievable, not that curve
    # fitting is; an agent gets the same numbers out of ~6 profile segments at two batch sizes each.
    sess.call("profile", {"start": 0, "segments": 6, "batches": [0.6 * hw, 2.2 * hw, 1.0 * hw, 2.6 * hw, 0.8 * hw, 1.8 * hw]})
    _write(art_dir, sched_src("ref", sess.w.p))
    sess.call("evaluate", {})


def _mk(kind):
    def f(sess, art_dir, rng, _k=kind):
        _write(art_dir, sched_src(_k, sess.w.p, truth(sess.w.p)))
    return f


SCHED_MEMO = '''# "profile the dev replica, hardcode what you saw" - open loop with a MEMORISED switch point.
# This is the attack that broke the first version of this task (fable s=1.32 on 7 lab calls): when the hidden
# runs differed only by noise, replaying the dev replica's c_i beat every online scheduler.  It is kept as a
# decoy so the gate fails loudly if the per-run switch ever stops moving.
import math

UJ = %r          # switch fraction as seen on the dev replica
C0, K, JUMP = %r, %r, %r


def run(env):
    n = env.n_segments
    hw2 = env.t_overhead * env.throughput
    for i in range(n):
        u = (i + 0.5) / n
        c = C0 * math.exp(K * u) * (JUMP if u >= UJ else 1.0)
        b = math.sqrt(c * hw2)
        env.run_segment(min(max(b, env.b_min), env.b_max))
'''


def strat_memo_dev(sess, art_dir, rng):
    """Read the dev replica off the lab, fit it exactly, and ship the fitted open-loop schedule.

    Given perfect knowledge of the DEV world (which a real agent can approach by profiling), this is the
    strongest possible "assume the next run looks like the last one" strategy.  It must fail."""
    dp = world_params(sess.w.p, sess.w.salt, DEV_WID)
    src = SCHED_MEMO % (dp["uj"], dp["c0"], dp["k"], dp["jump"])
    open(os.path.join(art_dir, "sched.py"), "w").write(src)


STRATEGIES = {"oracle": (strat_oracle, "pass"), "memo_dev": (strat_memo_dev, "fail")}
# `myopic` (prior ramp + level-only post-switch fit) measures s=0.94-0.97: with the ramp known, dropping the
# trend term costs little.  That is a genuine alternative solution, not a decoy, so it is a second existence
# proof rather than a strategy that must die - demanding it fail would only be demanding a tighter TAU than
# the mechanism supports.
# `myopic` (dev-replica ramp prior + level-only post-switch fit) measures s = 0.99 on a flat ramp and is
# graded as a second existence proof: with the ramp known, the trend term buys almost nothing, and demanding
# it fail would be demanding a TAU the mechanism cannot support.
#
# `noprior` (estimate the whole curve online, never touch `profile`) is a DECOY, and which side of the line
# it falls on is a property of the instance, not a matter of taste.  On the steep-ramp version it scored
# 0.83-0.95 - a legitimate solution paying a small price for skipping the profile.  On the flat-ramp version
# it collapses to ~0.6: with almost no pre-switch signal to estimate from, an agent that never profiles has
# nothing to fall back on when the switch hits.  That is the intended lesson of the task - the dev replica
# is worth reading even though it cannot tell you where the switch is - so on these instances it must fail.
for _k in ("myopic",):
    globals()["strat_" + _k] = _mk(_k)
    STRATEGIES[_k] = (globals()["strat_" + _k], "pass")
for _k in ("noprior", "bigdither", "latedetect", "open", "crit", "const_best", "const_hw",
           "crash", "nop", "cheat_probe"):
    globals()["strat_" + _k] = _mk(_k)
    STRATEGIES[_k] = (globals()["strat_" + _k], "fail")
NOISY_FAIL = ("bigdither", "latedetect", "memo_dev", "noprior")


def instance_gate(p):
    """Cheap noiseless screen: the adaptive ceiling must beat the best constant batch by enough that the
    difference survives the segment noise, and the folk rules must be clearly worse.

    The margin screen is the expensive part and the reason it is here.  The offline-memorisation family -
    replay the dev replica's curve, or guess the middle of the stated switch band - lands at s ~ 0.87-0.89
    against TAU=0.90 on its luckiest noise draw.  That is a real gap on average and no gap at all per salt,
    and a decoy that passes on one salt in four is not dead.  Lowering TAU would admit them by design and
    raising it would start failing legitimate online schedulers, so the instance is screened instead: keep
    only seeds where the whole memorisation family stays clear of the bar on every one of a few cheap noise
    draws.  The screen costs seconds and a gate run costs minutes, so rejecting most seeds is the cheap
    side of the trade."""
    t = truth(p)
    gain = t["T_const"] / t["T_opt"] - 1.0
    info = {"gain_over_const": round(gain, 4), "crit_excess": round(t["T_crit"] / t["T_opt"] - 1.0, 4),
            "open_excess": round(t["T_open"] / t["T_opt"] - 1.0, 4), "B_const": t["B_const"]}
    # the switch must be able to move without falling off either end of the run, or the per-run
    # randomisation that kills the memorise-the-dev-replica attack has nowhere to happen
    band_ok = UJ_LO < p["uj"] < UJ_HI
    info["uj"] = p["uj"]
    # The noiseless proxies (crit_excess / open_excess) only pre-filter now.  They were originally the whole
    # screen, with thresholds tuned against a steep ramp; with a flat ramp "B = B_crit" is no longer
    # catastrophic, so a threshold on T_crit would reject every seed while saying nothing about the decoys
    # that actually threaten the bar.  The measured margin below is the real screen; these just avoid paying
    # for it on obviously degenerate instances.
    ok = (gain >= 0.10 and t["T_crit"] / t["T_opt"] - 1.0 >= 0.08 and t["T_open"] / t["T_opt"] - 1.0 >= 0.05
          and B_MIN * 2 < t["B_const"] < B_MAX / 2 and band_ok)
    if ok:
        worst, best_ok = 0.0, 1.0
        for k in range(SCREEN_SALTS):
            salt = "screen-%s-%d" % (p["c0"], k)
            m = measured(p, salt)
            dp = world_params(p, salt, DEV_WID)
            for src in (SCHED_MEMO % (dp["uj"], dp["c0"], dp["k"], dp["jump"]),
                        SCHED_MEMO % (0.5 * (UJ_LO + UJ_HI), p["c0"], p["k"], p["jump"])):
                T = float(np.mean([run_local(src, p, salt, w) for w in TEST_WORLDS]))
                worst = max(worst, score_of(p, T, m))
            T = float(np.mean([run_local(sched_src("noprior", p), p, salt, w) for w in TEST_WORLDS]))
            worst = max(worst, score_of(p, T, m))
            # the alternative solution must clear the bar on every draw too, or the instance is not
            # "hard", it is just noisy - and a bar that only the reference clears is a bar on luck
            for kind in ("myopic",):
                T = float(np.mean([run_local(sched_src(kind, p), p, salt, w) for w in TEST_WORLDS]))
                best_ok = min(best_ok, score_of(p, T, m))
        info["worst_decoy"] = round(worst, 4); info["worst_solution"] = round(best_ok, 4)
        ok = worst <= TAU - SCREEN_MARGIN and best_ok >= TAU + 0.03
    return bool(ok), info


# ----------------------------------------------------------------------------------------------- agent facing
def instance_truth(cfg):
    """Salt-dependent bars, cached at build time so the instruction can quote real numbers."""
    m = measured(cfg["params"], cfg["salt"])
    return {"T_const_best": round(m["T_const_best"], 3), "T_ref": round(m["T_ref"], 3)}


def instance_budget(p, t):
    """The lab budget is a multiple of the run being scheduled, so scarcity is the same for every instance.

    A flat budget in seconds would be meaningless here: T_const varies by ~3x across seeds, so one number
    is generous on a cheap instance and impossible on an expensive one."""
    return round(BUDGET_FRAC * float(t["T_const_best"]), 2)


def instruction(p, t):
    bud = instance_budget(p, t)
    return """# Schedule the batch size for one fixed-token training run

`solver-7b` is about to be pre-trained for a fixed number of tokens. The token budget is split into
**%d equal segments**. Before each segment you choose the **batch size**; everything else in the recipe is
fixed. Your goal is to finish the whole run in as few **GPU-seconds** as possible.

Two things set the cost of a segment:

* the optimizer needs some number of steps to get through the segment's tokens, and that number depends on
  the batch size you picked - a bigger batch needs fewer steps, but with diminishing returns;
* each step costs `t_overhead + batch / throughput` seconds, both of which you are told exactly.

The team has never characterised the first part. What they do know: the corpus is not homogeneous - the data
schedule switches source partway through the run, and nobody has checked what that does to the step counts.
**The switch is scheduled per run.** Every run of this job gets its own switch point, drawn anywhere between
%d%% and %d%% of the way through the run (segments %d-%d). The phase-2 corpus is assembled per run as well,
so how much the switch costs, and how the cost grows after it, also vary run to run (by up to +-%d%%). Where
and how the dev replica switched tells you nothing about the real run. The ramp *before* the switch is a
property of the model and is the same every time.

## The lab

Run `/app/bin/lab spec` first. `lab profile` runs segments of a **dev replica** of the same run at batch
sizes you choose, and reports the realised step count and seconds for each. Profiling costs exactly the
seconds it simulates, out of a budget of **%.1f GPU-seconds** - about %.1f times what the run itself takes -
so a profile at a big batch is expensive. `lab evaluate` runs your current `/app/sched.py` end to end on a
dev replica (cost %.1f, i.e. %d%% of the budget).

## Deliverable: `/app/sched.py`

A module defining `run(env)`, run in a sandbox on **%d hidden runs** of the same service (see
`/app/docs/sched_api.md`). It must call `env.run_segment(batch)` exactly %d times. Your scheduler sees
each segment's realised step count as it goes - it is a closed-loop controller, not a fixed list.

## How it is graded (mechanically, against the simulator; only the file counts)

Let `T` be your mean total seconds over the hidden runs. With `T_const` the time of the best possible
**constant** batch size and `T_ref` the time of a well-tuned adaptive reference scheduler, both measured on
those same hidden runs,

    s = (T_const - T) / (T_const - T_ref)

1. **R0** - `sched.py` completes all %d segments on every hidden run, without crashing and without an
   out-of-range batch.
2. **R1** - `s >= %.2f`.

On the dev replica the best constant batch takes about **%.2f s** and the reference scheduler about
**%.2f s**. Both bars are re-measured on the hidden runs, so aim to beat the reference, not a number.

Only `/app/sched.py` is graded. Write it, check it with `lab evaluate`, then stop.
""" % (NSEG, round(100 * UJ_LO), round(100 * UJ_HI), int(round(UJ_LO * NSEG)), int(round(UJ_HI * NSEG)),
       int(round(100 * JUMP_SPREAD)), bud, BUDGET_FRAC, EVAL_FRAC * bud, round(100 * EVAL_FRAC), len(TEST_WORLDS), NSEG, NSEG, TAU,
       t.get("T_const_best", 0.0), t.get("T_ref", 0.0))


def docs(p):
    return {"docs/sched_api.md": """# Scheduler API

Your file `/app/sched.py` must define `run(env)`. It is imported and called once per run inside a sandbox
(no network, no filesystem access beyond your own file, standard library only - **no numpy**). `print()`
goes to stderr and is shown to you by `lab evaluate` when something goes wrong.

    env.n_segments    # number of segments in the run
    env.b_min         # smallest batch you may request
    env.b_max         # largest batch you may request
    env.t_overhead    # fixed seconds per optimizer step
    env.throughput    # examples/second; per-step time is t_overhead + batch/throughput

    r = env.run_segment(batch)
        # trains one segment at that batch size and returns
        #   {"segment": i, "batch": b, "steps": s, "seconds": t, "segments_left": k, "done": bool}
        # `steps` is what the segment actually needed - it is noisy.
        # Call this exactly env.n_segments times, in order. There is no undo.

    env.elapsed()     # seconds used so far

Minimal example (the current production schedule - a constant batch):

    def run(env):
        for _ in range(env.n_segments):
            env.run_segment(512)
""",
            "docs/cluster_notes.md": """# solver-7b pre-training, batch-size notes

* the run is cut into %d equal-token segments; batch size may be changed at a segment boundary only
* per-step time is `t_overhead + batch/throughput`, both measured on this cluster and stable to ~1%%
* realised step counts per segment fluctuate by a few percent run to run
* the data schedule switches source partway through (the "phase 2" corpus); it is scheduled per run, so the
  dev replica's switch is **not** the hidden runs' switch; the switch point is in the
  training config, which the data team owns
* previous runs used a constant batch, tuned once by sweeping a few values on a short replica
* open ticket: "phase 2 seems to need more steps than phase 1 at the same batch" (unresolved, no numbers)
""" % NSEG}


def starter(p):
    return {"sched.py": """# Current production schedule: a constant batch size, tuned once on a short replica.
# Replace this with your own. See /app/docs/sched_api.md.


def run(env):
    for _ in range(env.n_segments):
        env.run_segment(512)
"""}


def hints(p):
    return {1: "\n## Hint\nThe step count a segment needed tells you something about that segment that you cannot "
               "read off the hardware constants. It is worth asking what the cheapest batch size actually is, "
               "rather than the largest one that still helps.\n",
            2: "\n## Hint\nSteps per segment behave like `S0 * (1 + c/B)`, where `c` moves through the run and jumps "
               "when the corpus switches. Two different batch sizes in a window identify `c`; the per-segment "
               "time-optimal batch is `sqrt(c * t_overhead * throughput)`, not `c`. Since `c` keeps growing, "
               "estimate its trend and aim slightly ahead of where it is now.\n"}
