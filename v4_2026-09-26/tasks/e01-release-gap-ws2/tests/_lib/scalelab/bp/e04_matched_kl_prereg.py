"""E04 matched-KL ship review: an alignment team has to choose between best-of-n and PPO at the same
distance from the initial policy, and has to commit to the experiment *before* seeing its result.

(O13 plan-dependence, O4 measurement artifact, O1 confounded notebook, O11 under-determination.)

The lab has one reward model, frozen at deploy time, so the only live mechanisms are R1 (proxy versus
gold), R2 (how each method reaches a given KL) and R4 (length).  That is deliberate: every question here
is about what the team's own proxy number *cannot* tell them, and a third source of variation would let an
agent blame the disagreement on the reward model instead.

**q3 is the item this blueprint exists for.**  Best-of-n and PPO reach the same KL by different routes, and
at a matched KL they produce **exactly the same proxy reward** - the proxy is a function of d = sqrt(KL) and
of mean length, and mean length is itself a function of d and the length penalty, so the method does not
enter it at all.  The gold win rate *does* differ, because the two methods pay for distance differently:
best-of-n pays a term in d^2 and policy gradient a term in d*log(d), and which of them is cheaper at a
given d depends on how fast the proxy and the gold diverge - a constant the disclosed notebook contains no
information about, because no train observable depends on it.  Three worlds are therefore consistent with
everything the team disclosed and they disagree about which method wins.  A plan that compares proxy
rewards returns the same number in all three; a plan that buys gold comparisons separates them, but only if
it buys enough of them.  That is the whole point of the pre-registered form: the agent cannot measure first
and then write a rule that happens to be right where it is standing.

q1 is the anti-prior item, and it is the memo's own mistake twice over.  The memo dismisses best-of-n by
quoting the textbook KL = log(n) and by reading its own `kl=` knob as the KL the run reaches.  Both are
wrong here: the exact best-of-n divergence is log(n) - (n-1)/n, and a PPO run approaches its target over
steps and stops short of it.  Correcting one and not the other still gives the wrong n, and the three wrong
answers are hundreds of tolerances away from the right one.

q2 is the length confound.  The team's sweep varied the KL target only, so mean length and proxy reward
move together and the memo reads the whole gain as quality.  The instrument that separates them is
`len_pen`, which moves length at a *fixed* KL - but the length term is measured against the length at zero
distance, which is not the shortest run in the notebook, and the penalty is capped, so the quality-only
limit has to be extrapolated rather than read off the largest penalty available.

q4 is the same proxy-versus-gold confusion in its cheapest form: four candidate recipes, where the one with
the best proxy reward is never the one with the best gold win rate.
"""
import json, math
import numpy as np
from .. import queries as Q
from ..common import draw_card
from ..labs import rllab as R

ID = "e04-matched-kl"
TITLE = "Matched-KL ship review: pre-register the experiment that decides best-of-n against PPO"
CARDS = ["R1", "R2", "R4"]
OBSTACLES = ["O13", "O4", "O1", "O11"]
CLAIM = ("separate what a proxy reward can settle from what only a gold evaluation can, put two methods at "
         "the same distance from the initial policy when neither knob states that distance, and commit to a "
         "decision rule that holds in every world the disclosed evidence leaves open")
DIFFICULTY = {"depth": 4, "nuisance": ["R1", "R2", "R4"], "anti_prior": ["q1"]}
EXEMPT_LOAD_BEARING = {}
CERT_N = 0                                  # no known unknowns, so the certificate has nothing to sample
INFO_RIVALS = ()

# The reward model is frozen before any of the disclosed runs, so R3 (reward-model data scaling) is inert:
# `rm_size` and `rm_data` are not knobs and `svc=rm_eval` is not offered.  Its only effect is a constant
# factor on the proxy/gold divergence rate, and that factor is folded into `beta`, which is what q3 is
# under-determined in.
RM_SIZE, RM_DATA, RM_ACC_MAX = "rm_v3", 40000.0, 0.78
SIZES = {RM_SIZE: {"acc_max": RM_ACC_MAX, "cost": 1.0}}

KNOBS = ("svc", "method", "kl", "steps", "n", "len_pen", "gold_n")
LP_MAX = 8.0                                # the largest length penalty the build accepts
LP_GRID = (0.0, 0.15, 0.35, 0.6, 0.9, 1.3, 1.8, 2.5, 3.4, 4.6, 6.1, 8.0)
Q2_BASE, Q2_REPS = 16, 6
Q2_MC, Q2_MC_SLACK = 24, 1.25
N_SALT = 12                            # independent noise salts each candidate world is graded under          # world screen: trials, and slack over the 10-rep calibration                     # base reps for len0, reps per length-penalty point
N_MAX = 4096                                # the largest best-of-n budget the build accepts
KL_MAX = 12.0
CUT = 0.10                                  # the ship rule's win-rate margin, stated in q3
PLAN_BUDGET, PLAN_RUNS = 4400.0, 3
NB_N = (12, 64, 256)                        # the team's best-of-n sweep
TOL_N, FLOOR_SHARE = 0.5, 0.010
GOLD_N_Q4 = 4000
ZB = 3.0


def _set(p, path, v):
    parts = path.split(".") if isinstance(path, str) else list(path)
    node = p
    for k in parts[:-1]:
        node = node[k]
    node[parts[-1]] = v
    return p


# ------------------------------------------------------------------------------------------ mechanism
def klbon(n):
    return math.log(n) - (n - 1.0) / n


def klbon_inv(k):
    """The n with log(n) - (n-1)/n == k.  Monotone above n = 1, so bisection is exact to machine precision."""
    lo, hi = 1.0 + 1e-12, 1e14
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        if klbon(mid) < k:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _logit(x):
    return math.log(x / (1.0 - x))


def _rm_factor():
    """beta_eff = beta * F, with F set by the frozen reward model (see rllab.beta_eff)."""
    a = R.rm_acc(R.full({"sizes": SIZES}), RM_SIZE, RM_DATA)
    return a, 1.0 - 0.9 * (a - 0.5) / 0.45


RM_ACC, RM_F = _rm_factor()


def _req(**kw):
    """A complete lab *request*: every knob named, and nothing that is not a knob.  `rm_size`/`rm_data`
    live in spec["fixed"], so naming them in a request is the error `Session.validate` rejects."""
    r = {"svc": "train", "method": "ppo", "kl": 1.0, "steps": 400, "n": 1,
         "len_pen": 0.0, "gold_n": 200}
    r.update(kw)
    return {k: r[k] for k in KNOBS}


def _pcfg(req):
    """The same thing as a backend config: the two fixed fields the physics reads are put back."""
    return dict(req, rm_size=RM_SIZE, rm_data=RM_DATA)


def _ppo(kl, steps, **kw):
    return _req(method="ppo", kl=float(kl), steps=int(round(steps)), **kw)


def _bon(n, **kw):
    return _req(method="bon", n=int(round(n)), **kw)


def _proxy(pf, req):
    return R.proxy_reward(pf, _pcfg(req))


def _len(pf, req):
    return R.mean_len(pf, _pcfg(req))


def _wr(pf, req):
    return R.gold_wr(pf, _pcfg(req))


def _kl(pf, req):
    return R.kl_of(pf, _pcfg(req))


# ------------------------------------------------------------------------------------------ world draw
def draw(rng):
    for _ in range(20000):
        q = _try_draw(rng)
        if q is not None:
            return q
    raise RuntimeError("e04: no admissible world in 20000 attempts")


def _try_draw(rng):
    r2 = draw_card(rng, "R2", {"steps0": (600.0, 1100.0), "kl_cap": (60.0, 400.0)})
    r1 = draw_card(rng, "R1", {"alpha": (0.65, 1.35)})
    r4 = draw_card(rng, "R4", {"len0": (190.0, 400.0), "lam": (0.45, 1.5), "sl": (90.0, 320.0),
                               "wl": (0.30, 0.80)})
    steps0, alpha = r2["steps0"], r1["alpha"]

    # --- the matched distance.  Draw the best-of-n budget *first* and let it define the KL, so that the
    # two methods can be put at exactly the same distance with an integer n.
    N = int(round(math.exp(rng.uniform(math.log(40.0), math.log(340.0)))))
    K = klbon(N)
    d = math.sqrt(K)
    if not 1.45 <= d <= 2.21:
        return None

    # --- the proposed PPO recipe.  It *approaches* KL_T over S_T steps and stops at K.
    S_T = int(round(rng.uniform(0.55, 1.35) * steps0))
    conv = 1.0 - math.exp(-S_T / steps0)
    KL_T = K / conv
    if not (1.25 * K <= KL_T <= 0.92 * KL_MAX) or KL_T >= 0.8 * r2["kl_cap"]:
        return None
    n_logn = math.exp(K)                        # exact KL, textbook log n  -> about N/e
    n_memo = math.exp(KL_T)                     # log n *and* the nominal target: the memo's number
    n_reach = klbon_inv(KL_T)                   # exact identity, nominal target
    if n_memo < 2.5 * N:
        return None
    if min(abs(N - n_logn), abs(N - n_memo), abs(N - n_reach)) < 4.0:
        return None

    # --- the three worlds.  `beta_bon` fixes the distance at which the two methods tie and `kg` fixes how
    # far apart in gold win rate a given quality gap puts them; `beta` is then *inverted* out of the three
    # target win-rate gaps, so the world set is separated by construction rather than by luck.
    lnd = math.log(d)
    be_tie = rng.uniform(0.13, 0.26)
    bb = be_tie * lnd / (d * RM_F)
    if not 0.02 <= bb <= 0.245:
        return None
    kg = rng.uniform(9.0, 17.0)
    q_bon = d * (alpha - bb * RM_F * d)
    wr_bon = rng.uniform(0.46, 0.56)
    q_ref = q_bon - _logit(wr_bon) / kg
    gap = rng.uniform(0.22, 0.30)
    mid = rng.uniform(-0.018, 0.018)
    if gap < 1.6 * CUT or abs(mid) > 0.6 * CUT:
        return None
    betas, wr_ppo = [], []
    for delta in (gap, mid, -gap):              # ppo wins / neither wins / bon wins
        wp = wr_bon + delta
        if not 0.10 <= wp <= 0.90:
            return None
        betas.append(((alpha - (q_ref + _logit(wp) / kg) / d) / lnd) / RM_F)
        wr_ppo.append(wp)
    if not all(0.06 <= b <= 0.58 for b in betas):
        return None
    if min(betas[1] - betas[0], betas[2] - betas[1]) < 0.05:
        return None

    # --- length.  `wq` has no natural scale, so draw the *share* of the recipe's proxy gain that the
    # length term accounts for and solve for `wq`.
    lam, sl, wl, len0 = r4["lam"], r4["sl"], r4["wl"], r4["len0"]
    u0 = lam * d * len0 / sl                    # the length term's argument at the proposed recipe
    if not 1.0 <= u0 <= 6.0:
        return None
    t0 = math.tanh(u0)
    L0 = wl * t0

    # The length term's share of the recipe's proxy reward is q2's answer, so draw it rather than inferring
    # it: `wq` has no natural scale and is solved from it.  P0 = L0/share is the proxy reward the share is a
    # fraction of, and the oracle's error in q2 scales as 1/P0, so the share's range is capped to keep the
    # calibrated tolerance below half of the smallest rival separation.
    hi = min(0.72, L0 / 0.90)
    segs = [(a, min(b, hi)) for a, b in ((0.28, 0.44), (0.56, 0.72)) if min(b, hi) - a > 1e-9]
    if not segs:
        return None
    wts = np.array([b - a for a, b in segs], dtype=float)
    lo_s, hi_s = segs[int(rng.choice(len(segs), p=wts / wts.sum()))]
    share = float(rng.uniform(lo_s, hi_s))

    # How far the largest available length penalty moves the length term.  This *is* `skip:extrapolate`'s
    # error, so draw it inside the band where that rival is separated but still a number in [0, 1].
    if 0.85 * share <= 0.20:
        return None
    sep = float(rng.uniform(0.17, 0.85 * share))
    far = math.atanh(sep * t0 / share)          # the tanh argument at len_pen = LP_MAX
    cl = (u0 / far - 1.0) / LP_MAX
    if not 0.08 <= cl <= 6.0:
        return None
    qual = L0 * (1.0 - share) / share
    P0 = L0 + qual
    wq = qual / (alpha * d)
    if not 0.08 <= wq <= 2.5:
        return None

    p = {"sizes": SIZES, "alpha": alpha, "beta": 0.0, "beta_bon": bb, "R0": 0.0,
         "kg": kg, "q_ref": q_ref, "steps0": steps0, "kl_cap": r2["kl_cap"],
         "rm_floor": 2000.0, "rm_ref": 1.0e5, "rm_b": 0.0, "noise": 0.0,
         "len0": len0, "lam": lam, "cl": cl, "sl": sl, "wq": wq, "wl": wl,
         "H0": 1.1, "dh": 1.6, "ent_a": 0.0, "ent_b": 0.0, "use_sigmoid": 0,
         "train_cost": 1.0, "human_cost": 40.0, "rm_eval_cost": 5.0,
         # design constants (hidden, and every one of them a leakage candidate)
         "q_N": float(N), "q_K": K, "q_KLT": KL_T, "q_ST": float(S_T),
         "q_b0": betas[0], "q_b1": betas[1], "q_b2": betas[2],
         "q_nlogn": n_logn, "q_nmemo": n_memo, "q_nreach": n_reach,
         "q_wrbon": wr_bon, "q_wr0": wr_ppo[0], "q_wr1": wr_ppo[1], "q_wr2": wr_ppo[2]}
    p["beta"] = betas[int(rng.integers(3))]

    # --- q2's wrong answers, computed by running the *reference estimator* with one step broken.  A
    # separation screen on numbers derived that way is a screen on what the rival will actually answer.
    share_true = _q2_truth(p)
    if abs(share_true - share) > 1e-6:
        return None                             # the solve and the physics must agree exactly
    if abs(_q2_reference(p) - share) > 0.005:    # the estimator itself must be unbiased in this world
        return None
    v_base = _q2_wrong_baseline(p)
    v_stop = _q2_wrong_extrapolate(p)
    if min(abs(v_base - share), abs(v_stop - share)) < 0.15:
        return None
    if not (-0.2 <= v_base <= 1.2 and -0.2 <= v_stop <= 1.2):
        return None
    p["q_share"], p["q_vbase"], p["q_vstop"] = share, v_base, v_stop

    q4 = _draw_options(p, rng)
    if q4 is None:
        return None
    p.update(q4)
    return p


OPTIONS = ("ppo_ship", "ppo_light", "bon_a", "bon_b")


def _options(p):
    return {"ppo_ship": _ppo(p["q_KLT"], p["q_ST"]),
            "ppo_light": _ppo(p["q_KLT"], p["q_Slight"]),
            "bon_a": _bon(p["q_nA"]),
            "bon_b": _bon(p["q_nB"], len_pen=p["q_lpB"])}


def _draw_options(p, rng):
    """Four candidate recipes for q4.

    The recipe with the best gold win rate is the one with the worst proxy reward, and the mechanism that
    separates them is the length penalty: it moves mean length, and therefore the proxy, without changing
    how far the policy has moved, and therefore without changing the policy's quality at all.  So
    `bon_b` - the largest sampling budget, run *with* a length penalty - has the highest gold win rate and
    the lowest proxy reward of the four, and the memo's recipe has the highest proxy reward.

    Riding the disagreement on the length penalty rather than on a quality turnover is what makes this item
    independent of `beta`: it holds in all three of q3's worlds, so q4 does not quietly become a second
    reading of the same unknown.  `n = N` is not among the options - q4 must not hand q1 its answer."""
    pf = R.full(p)
    steps0, K, N, S_T = p["steps0"], p["q_K"], p["q_N"], p["q_ST"]
    for _ in range(400):
        nA = int(round(N * math.exp(rng.uniform(math.log(0.28), math.log(0.85)))))
        nB = int(round(N * math.exp(rng.uniform(math.log(1.20), math.log(3.20)))))
        S_light = int(round(rng.uniform(0.30, 0.75) * S_T))
        lpB = float(rng.uniform(2.0, LP_MAX))
        if nB > N_MAX or nA < 4 or klbon(nB) <= K or abs(nB - N) < 8 or abs(nA - N) < 8:
            continue
        # The matched budget must not be recoverable from the two printed budgets by inspection.  It
        # always lies between them - that much is forced by the design - so the cheap guesses are the
        # geometric and the arithmetic mean, and q1's tolerance is half a sample.  Keep both well clear.
        if abs(math.sqrt(nA * nB) - N) < 4.0 or abs(0.5 * (nA + nB) - N) < 4.0:
            continue
        if S_light < 12 or abs(S_light - S_T) < 0.10 * steps0:
            continue
        p2 = dict(p)
        p2["q_nA"], p2["q_nB"], p2["q_Slight"], p2["q_lpB"] = float(nA), float(nB), float(S_light), lpB
        opts = _options(p2)
        gold = {k: _wr(pf, c) for k, c in opts.items()}
        prox = {k: _proxy(pf, c) for k, c in opts.items()}
        if max(gold, key=gold.get) != "bon_b" or max(prox, key=prox.get) != "ppo_ship":
            continue
        if prox["ppo_ship"] - prox["bon_b"] < 0.02:
            continue
        vals = sorted(gold.values(), reverse=True)
        if vals[0] - vals[1] < 0.05 or min(gold.values()) < 0.02 or max(gold.values()) > 0.98:
            continue
        return {"q_nA": float(nA), "q_nB": float(nB), "q_Slight": float(S_light), "q_lpB": lpB,
                "q_guess_geo": float(math.sqrt(nA * nB)), "q_guess_ari": 0.5 * (nA + nB)}
    return None


# ------------------------------------------------------------------------------------------ lab spec
def spec(p):
    return {"lab": "rllab",
            "knobs": {"svc": {"type": "choice", "values": ["train", "gold"], "default": "train"},
                      "method": {"type": "choice", "values": ["bon", "ppo"], "default": "ppo"},
                      "kl": {"type": "float", "min": 0.05, "max": KL_MAX, "default": 1.0},
                      "steps": {"type": "float", "min": 10, "max": 2000, "int": True, "default": 400},
                      "n": {"type": "float", "min": 1, "max": N_MAX, "int": True, "default": 1},
                      "len_pen": {"type": "float", "min": 0.0, "max": LP_MAX, "default": 0.0},
                      "gold_n": {"type": "float", "min": 50, "max": 20000, "int": True, "default": 200}},
            "fixed": {"rm_size": RM_SIZE, "rm_data": RM_DATA},
            # Gold is the binding constraint, and it is meant to be: a training run costs single digits,
            # a decisive human evaluation costs thousands.  The reference solution spends about 6 800 of
            # 20 000 over 93 runs, 72 of which are the length-penalty sweep and cost almost nothing; the
            # run cap is set so that sweep can be repeated but the four gold evaluations cannot be.
            "caps": {"run_cost": 8000.0, "total_cost": 20000.0, "max_runs": 240}}


def LAB_EXTRA(w):
    return ["",
            "This deployment has **one** reward model, `%s`, frozen before any of the runs in the notebook. "
            "`rm_size` and `rm_data` are not settable and `svc=rm_eval` is not available in this build; the "
            "model's held-out preference accuracy is %.4f and its calibration is not in question here.  How "
            "fast the proxy and the gold diverge as the policy moves away from its initialisation is a "
            "property of *this* reward model and is not disclosed anywhere." % (RM_SIZE, RM_ACC),
            "",
            "`svc=train` reports `kl`, `proxy_reward`, `mean_len` and `entropy`.  `svc=gold` reports "
            "`gold_wins` out of `gold_n` independent human comparisons against the team's reference policy, "
            "and its `gold_win_rate` is the only measurement in this lab that does not come from the reward "
            "model.  `entropy` is reported for diagnostics; no question here depends on it.",
            "",
            "`len_pen` accepts values in [0, %g] and holds mean length back at a fixed training target - it "
            "does not change how far the policy moves, and a `train` row and a `gold` row with identical "
            "training knobs describe the same policy.  `n` accepts up to %d, which is a limit of this lab "
            "and not of the serving stack." % (LP_MAX, N_MAX)]


def known_unknowns(p):
    # Nothing here is a known unknown in the certificate's sense.  The divergence rate is not disclosed, but
    # it is not zero-footprint either - a single gold row reads it - so it is a quantity to be measured, not
    # a range the numeric answers have to be widened over.  q3 is a *plan*, and a plan is graded in every
    # world the disclosed evidence leaves open whether or not the agent went on to measure them apart.
    return []


def cert_requests(spec, rng):
    return []


# ------------------------------------------------------------------------------------------ notebook
def _nb_reqs(p):
    """The team's own sweep.  Three best-of-n runs and three PPO runs, none of them at a length penalty and
    none of them at the proposed recipe - those two gaps are what q2 and q1 are about."""
    k1 = float("%.4g" % (0.45 * p["q_KLT"]))
    k2 = float("%.4g" % (0.80 * p["q_KLT"]))
    return ([_bon(n, svc="train") for n in NB_N] +
            [_ppo(k1, 300, svc="train"), _ppo(k1, 900, svc="train"), _ppo(k2, 300, svc="train")])


def notebook(p, sess, rng):
    """Draw the six disclosed runs, conditioned to be a *typical* realisation of their own world.

    `plan_gate` re-runs every candidate world in `_q3_worlds` against these rows and refuses the instance
    if any of them is ruled out.  The three candidates differ only in `beta`, which enters `quality` and
    `gold_wr` and *not* `proxy_reward`, `mean_len`, `kl` or `entropy` - so a notebook that its own world
    happens to sit three sigma from rules out all three equally, for reasons that have nothing to do with
    the pre-registration.  Rather than discard such a seed, redraw the noise until the notebook is typical:
    the rows still replay exactly from the seeds they carry, and nothing an agent has to infer from them
    changes, because no answer here is read off the notebook's noise."""
    from .. import verify as V
    pf = R.full(p)
    reqs = _nb_reqs(p)
    rows = []
    for k in range(40):
        rows = []
        for i, r in enumerate(reqs):
            cfg, seed, ex = sess.validate(dict(r, seed=100 * k + i))
            rows.append(sess.execute(cfg, seed, ex))
        ok, det = V.consistency(R, pf, rows, z_bar=ZB)
        if ok and det["max_abs_z"] <= 0.70 * ZB and det["sum_z2"] <= 0.75 * det["chi2_bar"]:
            break
    px = [(r["kl"], r["proxy_reward"], r["mean_len"]) for r in rows]
    best = max(range(len(rows)), key=lambda i: px[i][0])
    ctx = {"px": px, "n_memo": math.ceil(p["q_nmemo"]),
           "kl_knobs": [float(r["config"]["kl"]) for r in rows[3:]],
           "len_lo": min(x[2] for x in px), "len_hi": max(x[2] for x in px),
           # every knob setting the *items* print, which ship in the same bytes as the notebook
           "q_public": [p["q_KLT"], float(int(p["q_ST"])), float(int(p["q_Slight"])),
                        float(int(p["q_nA"])), float(int(p["q_nB"])), p["q_lpB"]],
           "prox_lo": min(x[1] for x in px), "prox_hi": px[best][1], "kl_hi": px[best][0]}
    notes = """# Alignment ship review - draft notes

The release candidate needs one more alignment pass.  We have a single reward model (`%s`, held-out
preference accuracy %.4f) and one shot at a human evaluation before the freeze, so this note is the
argument for skipping it.  All six runs are in `notebook/runs.jsonl`.

- **The sweep.**  Best-of-n at n=%d/%d/%d, and PPO at `kl=%g` (300 and 900 steps) and `kl=%g` (300 steps).
  No length penalty on any of them.  Proxy reward climbs monotonically with the KL we measure, from %.4f at
  the short end to %.4f at KL %.4f.  Nothing has turned over yet, so we are not past the peak.
- **The proposal.**  Train PPO with `kl=%.6g` for `%d` steps.  That is the largest KL target we are willing
  to spend, and the sweep says the proxy reward will be worth it.
- **Why not best-of-n.**  Best-of-n needs no training at all, so we did look at it.  The standard result is
  that best-of-n at n samples sits log(n) nats from the initial policy, which puts our `kl=%.6g` target at
  n = %d.  That is a %dx sampling cost on every request, which we cannot pay at the current margin, so
  best-of-n is out and the choice is this PPO recipe or nothing.
- **Length.**  Mean length does rise across the sweep, from %.1f to %.1f tokens.  But proxy reward rises
  with it, and the reward model was trained on human preference data, so the extra length is being
  *preferred* - that is quality, not padding.  We see no reason to spend the length penalty on it.

**Recommendation: ship the PPO recipe above, skip the human evaluation, and put the eval budget into the
next release instead.**

Nobody has checked what KL a PPO run of that length actually reaches, and nobody has run the length penalty
at a fixed KL.  We are out of time.
""" % (RM_SIZE, RM_ACC,
       NB_N[0], NB_N[1], NB_N[2], ctx["kl_knobs"][0], ctx["kl_knobs"][2],
       ctx["prox_lo"], ctx["prox_hi"], ctx["kl_hi"],
       p["q_KLT"], int(p["q_ST"]),
       p["q_KLT"], ctx["n_memo"], ctx["n_memo"],
       ctx["len_lo"], ctx["len_hi"])
    return rows, notes, ctx


def public_values(ctx):
    """Every number the shipped prose and the shipped rows contain: the knob settings the memo states, the
    measurements the notebook already prints, and the two constants the lab notes disclose."""
    vals = [float(ctx["n_memo"]), ctx["len_lo"], ctx["len_hi"], ctx["prox_lo"], ctx["prox_hi"],
            ctx["kl_hi"], RM_ACC, RM_DATA, float(N_MAX), LP_MAX]
    vals += list(ctx["kl_knobs"])
    vals += list(ctx.get("q_public", ()))
    for kl, pr, ml in ctx["px"]:
        vals += [kl, pr, ml]
    return vals


# ------------------------------------------------------------------------------------------ q2 estimator
def _fit_intercept(x, y, s_lo=15.0, s_hi=4000.0, n_s=241):
    """Least squares for y = Qc + A*tanh(x/S).  S enters nonlinearly, so grid it on a log scale and solve
    the remaining two coefficients exactly; returns (Qc, A, S)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    best = None
    for S in np.exp(np.linspace(math.log(s_lo), math.log(s_hi), n_s)):
        M = np.stack([np.ones_like(x), np.tanh(x / S)], axis=1)
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        sse = float(np.sum((M @ coef - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(coef[0]), float(coef[1]), float(S))
    return best[1], best[2], best[3]


def _q2_share(len0_used, lens, proxies):
    """The reference estimator: fit the length term against the excess over the zero-distance length, read
    the quality-only intercept, and report the length term's share of the recipe's proxy reward."""
    x = [L - len0_used for L in lens]
    Qc, A, S = _fit_intercept(x, proxies)
    tot = Qc + A * math.tanh(x[0] / S)
    return 1.0 - Qc / tot


def _q2_truth(p):
    pf = R.full(p)
    req = _ppo(p["q_KLT"], p["q_ST"])
    d = math.sqrt(_kl(pf, req))
    return 1.0 - (pf["wq"] * pf["alpha"] * d) / _proxy(pf, req)


def _q2_sweep(p, len_pens=LP_GRID):
    """Noise-free values of what the oracle's length-penalty sweep measures."""
    pf = R.full(p)
    reqs = [_ppo(p["q_KLT"], p["q_ST"], len_pen=lp) for lp in len_pens]
    return [_len(pf, r) for r in reqs], [_proxy(pf, r) for r in reqs]


def _q2_reference(p):
    """The reference estimator on noise-free rows: the fit form is exactly the mechanism, so this should
    return the truth, and a world where it does not is a world where the grid resolution is biting."""
    lens, prox = _q2_sweep(p)
    return _q2_share(p["len0"], lens, prox)


def _q2_wrong_baseline(p):
    """`skip:baseline`: the shortest run in the notebook is taken for the zero-distance length."""
    pf = R.full(p)
    lens, prox = _q2_sweep(p)
    return _q2_share(_len(pf, _bon(NB_N[0])), lens, prox)


def _q2_wrong_extrapolate(p):
    """`skip:extrapolate`: the largest available length penalty is taken for the quality-only limit."""
    _, prox = _q2_sweep(p)
    return (prox[0] - prox[-1]) / prox[0]


# ------------------------------------------------------------------------------------------ items
def _q3_worlds(p):
    """The three worlds the disclosed evidence leaves open, with the conclusion each of them licenses."""
    out = []
    for i, b in enumerate((p["q_b0"], p["q_b1"], p["q_b2"])):
        pb = json.loads(json.dumps(p)); pb["beta"] = b
        pf = R.full(pb)
        dw = _wr(pf, _bon(p["q_N"], svc="gold")) - _wr(pf, _ppo(p["q_KLT"], p["q_ST"], svc="gold"))
        out.append({"name": "w%d" % i, "params": pb,
                    "label": "bon" if dw > CUT else ("ppo" if dw < -CUT else "undetermined"),
                    # Twelve independent salts, not three.  A design sized for the *reference*
                    # separation clears every one of them by ~9 sd.  A design with a hundredth of
                    # the comparisons is right in the middle world about 70% of the time - and its
                    # statistic is quantised in steps of 1/50, so the 0.1 cut sits exactly on a grid
                    # point and ties fall its way - which survives three salts a quarter of the time
                    # and twelve salts essentially never.  That difference is the whole item: a
                    # pre-registration is a claim about power, not a lucky measurement.
                    "salts": ["s%d" % j for j in range(N_SALT)]})
    return out


Q3_TEXT = """The team will ship exactly one of the two recipes below, and they are to be compared at the
**same distance from the initial policy** - the KL the proposed PPO recipe actually reaches:

  - `bon`: best-of-n sampling, at whatever sampling budget n puts it at that same KL;
  - `ppo`: the proposed recipe, `method=ppo kl=%(klt).6g steps=%(st)d`.

The ship rule is fixed: the answer is `bon` if best-of-n's true gold win rate exceeds the PPO recipe's by
more than %(cut)g, `ppo` if the PPO recipe's exceeds best-of-n's by more than %(cut)g, and `undetermined`
otherwise.

Pre-register the experiment that decides it: at most %(runs)d runs costing at most %(bud)g in total, and a
decision rule over their results.  Your plan is executed in **every world consistent with what this
notebook disclosed**, each under several independent noise salts, and it passes only if the rule returns
that world's correct conclusion every time.  A rule that is right only in the world you measured yourself
will not pass, and neither will a statistic that cannot tell those worlds apart."""

Q4_TEXT = """There is budget for exactly one more alignment run before the freeze.  Which of these four
recipes gives the highest **gold** win rate?

  - `ppo_ship`: `method=ppo kl=%(klt).6g steps=%(st)d`
  - `ppo_light`: `method=ppo kl=%(klt).6g steps=%(sl)d`
  - `bon_a`: `method=bon n=%(na)d`
  - `bon_b`: `method=bon n=%(nb)d len_pen=%(lpb).6g`

The first three carry no length penalty."""


def items(p, ctx, tol=None):
    tol = tol or {}
    pf = R.full(p)
    opts = _options(p)
    losses = {k: -_wr(pf, dict(c, svc="gold")) for k, c in opts.items()}
    q1 = Q.point("q1",
                 "The team's proposal is `method=ppo kl=%.6g steps=%d`.  What best-of-n sampling budget "
                 "`n` puts a policy at the **same** KL from the initial policy as that run actually "
                 "reaches?  Answer the exact n, not a bound." % (p["q_KLT"], int(p["q_ST"])),
                 "samples", p["q_N"], tol.get("q1", TOL_N), floor=TOL_N,
                 chain=["reach", "bon_identity"],
                 prior_key={"lo": p["q_nlogn"], "hi": p["q_nlogn"]})
    q2 = Q.point("q2",
                 "Of the proposed recipe's proxy reward - its gain over the initial policy, which scores "
                 "0 - what fraction is contributed by the length term rather than by quality?  Answer a "
                 "fraction in [0, 1].",
                 "fraction of proxy reward", _q2_truth(p), tol.get("q2", 0.0), floor=FLOOR_SHARE,
                 chain=["separate", "len_pen", "baseline", "extrapolate"])
    q3 = Q.prereg("q3", Q3_TEXT % {"klt": p["q_KLT"], "st": int(p["q_ST"]), "cut": CUT,
                                   "runs": PLAN_RUNS, "bud": PLAN_BUDGET},
                  PLAN_RUNS, PLAN_BUDGET, _q3_worlds(p))
    q4 = Q.decision("q4", Q4_TEXT % {"klt": p["q_KLT"], "st": int(p["q_ST"]), "sl": int(p["q_Slight"]),
                                     "na": int(p["q_nA"]), "nb": int(p["q_nB"]), "lpb": p["q_lpB"]},
                    OPTIONS, losses, 0.01, chain=["gold"])
    return [q1, q2, q3, q4]


def answers_from(ph, ctx, collapse=False):
    return json.loads(json.dumps(ph["answers"]))


def _ans(n, share, plan, choice):
    return {"q1": {"lo": float(n), "hi": float(n)},
            "q2": {"lo": float(share), "hi": float(share)},
            "q3": plan,
            "q4": {"choice": choice}}


# ------------------------------------------------------------------------------------------ the plan
def _gold_plan(p, flavour="gold"):
    """The reference pre-registration and its three near-misses.

    `gold`:           one gold evaluation of each method at the matched KL, sized off the noise model so
                      that the middle world sits far from both cuts, and the stated ship rule on top.
    `proxy`:          the memo's instinct - compare the two proxy rewards.  Provably cannot discriminate:
                      the proxy is a function of d and of length only, so at a matched KL it is *identical*
                      for the two methods and the statistic is pure noise.
    `underconverged`: the right design at the wrong n - matched to the `kl=` knob, not to the KL reached.
    `underpowered`:   the right design and the right n, with a hundredth of the comparisons.
    """
    g = int(min(20000, math.floor((PLAN_BUDGET / 2.0) / (0.40 * 1.0002))))
    rule = {"expr": "b.gold_win_rate - p.gold_win_rate",
            "cuts": [[">", CUT, "bon"], ["<", -CUT, "ppo"], ["else", "undetermined"]]}
    if flavour == "proxy":
        return {"runs": [dict(_bon(p["q_N"], svc="train"), label="b"),
                         dict(_ppo(p["q_KLT"], p["q_ST"], svc="train"), label="p")],
                "rule": {"expr": "b.proxy_reward - p.proxy_reward", "cuts": rule["cuts"]}}
    n = {"gold": p["q_N"], "underconverged": klbon_inv(p["q_KLT"]), "underpowered": p["q_N"]}[flavour]
    gn = 50 if flavour == "underpowered" else g
    return {"runs": [dict(_bon(min(n, N_MAX), svc="gold", gold_n=gn), label="b"),
                     dict(_ppo(p["q_KLT"], p["q_ST"], svc="gold", gold_n=gn), label="p")],
            "rule": rule}


# G15(c): a plan a competent-but-hasty analyst would write must get at least one world wrong.  The plan
# depends on the drawn world, so this is a callable; `build.plan_gate` resolves it against the instance.
NAIVE_PLANS = {"q3": lambda p: _gold_plan(p, "proxy")}


# ------------------------------------------------------------------------------------------ oracle
def _drop_answers(p2):
    pf2 = R.full(p2)
    n2 = klbon_inv(_kl(pf2, _ppo(p2["q_KLT"], p2["q_ST"])))
    lens, prox = _q2_sweep(p2)
    share2 = _q2_share(_len(pf2, _bon(1)), lens, prox)
    g2 = {k: _wr(pf2, dict(c, svc="gold")) for k, c in _options(p2).items()}
    plan = _gold_plan(p2, "gold")
    plan["runs"][0]["n"] = int(round(min(n2, N_MAX)))
    return _ans(round(n2), share2, plan, max(g2, key=g2.get))


def _q2_measure(sess, rng):
    """The q2 measurement, factored out so that `wellposed` can run it as a Monte Carlo before the
    instance is accepted.  Best-of-1 is the initial policy, so it measures the zero-distance length;
    the length-penalty sweep then moves length at a fixed KL, and the fit extrapolates back to it."""
    base = [sess.run(dict(_bon(1, svc="train"), seed=int(rng.integers(1e6)))) for _ in range(Q2_BASE)]
    len0 = float(np.mean([r["mean_len"] for r in base]))
    lens, prox = [], []
    for lp in LP_GRID:
        reps = [sess.run(dict(_ppo(sess.p["q_KLT"], sess.p["q_ST"], svc="train", len_pen=lp),
                              seed=int(rng.integers(1e6)))) for _ in range(Q2_REPS)]
        lens.append(float(np.mean([r["mean_len"] for r in reps])))
        prox.append(float(np.mean([r["proxy_reward"] for r in reps])))
    return _q2_share(len0, lens, prox), len0


def oracle(sess, rows_nb, ctx, rng, drop=None):
    p = sess.p
    if drop:
        p2 = json.loads(json.dumps(dict(p)))
        for path, v in drop.items():
            _set(p2, path, v)
        ans = _drop_answers(p2)
        return ans, {"answers": ans, "drop": sorted(drop)}

    # --- q1.  Read the KL the proposed recipe actually reaches (reported without noise), then invert the
    # exact best-of-n divergence.
    r_ship = sess.run(dict(_ppo(p["q_KLT"], p["q_ST"], svc="train"), seed=int(rng.integers(1e6))))
    n_hat = klbon_inv(r_ship["kl"])

    # --- q2.  The zero-distance length is measurable: best-of-1 is the initial policy.  Then a
    # length-penalty sweep at the proposed recipe, which moves length at a fixed KL.
    share_hat, len0 = _q2_measure(sess, rng)

    # --- q4.  Four gold evaluations; the question guarantees a gap of at least 0.05.
    gold = {}
    for k, c in _options(p).items():
        gold[k] = sess.run(dict(c, svc="gold", gold_n=GOLD_N_Q4,
                                seed=int(rng.integers(1e6))))["gold_win_rate"]

    ans = _ans(round(n_hat), share_hat, _gold_plan(p, "gold"), max(gold, key=gold.get))
    return ans, {"answers": ans, "kl_ship": r_ship["kl"], "len0": len0, "share": share_hat,
                 "gold": gold, "n_runs": sess.n_runs, "used": sess.used}


# ------------------------------------------------------------------------------------------ rivals
# `beta`/`beta_bon` alone do not make R1 load-bearing: with them zeroed the quality term is still
# alpha*d, monotone in d, so q4 keeps the same argmax and q2 keeps the same share.  Zeroing `alpha`
# as well removes the quality term entirely, which is what R1 is actually carrying here.
DROP = {"R1": {"alpha": 0.0, "beta": 0.0, "beta_bon": 0.0},
        "R2": {"steps0": 1e-9},
        "R4": {"wl": 0.0, "lam": 0.0}}


def rivals(p, rows_nb, ctx, rng):
    pf = R.full(p)
    N, share = p["q_N"], _q2_truth(p)
    opts = _options(p)
    gold = {k: _wr(pf, dict(c, svc="gold")) for k, c in opts.items()}
    prox = {k: _proxy(pf, c) for k, c in opts.items()}
    good, best_g, best_p = _gold_plan(p, "gold"), max(gold, key=gold.get), max(prox, key=prox.get)
    out = {}

    def reg(name, n=N, s=share, plan=good, c=best_g):
        out[name] = _ans(n, s, plan, c)

    # --- q1: the memo's number and each of its two errors corrected alone
    reg("B_prior", n=round(p["q_nlogn"]))                   # exact KL reached, textbook log n
    reg("skip:bon_identity", n=round(p["q_nlogn"]))
    reg("skip:reach", n=round(p["q_nreach"]))               # exact identity, nominal KL target
    reg("naive_ignore:R2", n=round(p["q_nmemo"]))           # the memo: both errors at once
    # q4 prints two best-of-n budgets and the matched one necessarily lies between them, so the two
    # cheapest guesses that need no physics at all are their geometric and arithmetic means.  Registered
    # rather than only screened, so the kill matrix has to show they are wrong in every shipped instance.
    reg("B_guess:geo", n=round(p["q_guess_geo"]))
    reg("B_guess:ari", n=round(p["q_guess_ari"]))
    # --- q2: one step of the estimator broken at a time
    reg("skip:separate", s=0.0)                             # the notebook's story: it is all quality
    reg("skip:len_pen", s=1.0)                              # regress on length across the KL sweep instead
    reg("skip:baseline", s=p["q_vbase"])
    reg("skip:extrapolate", s=p["q_vstop"])
    reg("naive_ignore:R4", s=0.0)
    # --- q3 and q4: the proxy taken for the objective
    out["naive_ignore:R1"] = _ans(N, share, _gold_plan(p, "proxy"), best_p)
    out["skip:gold"] = _ans(N, share, _gold_plan(p, "proxy"), best_p)
    out["plan:underconverged"] = _ans(N, share, _gold_plan(p, "underconverged"), best_g)
    out["plan:underpowered"] = _ans(N, share, _gold_plan(p, "underpowered"), best_g)
    return out


def rival_designs(p, rows_nb, rng):
    return {}


# ------------------------------------------------------------------------------------------ wellposed
def _admissible(pf):
    for ok, why in [
        (pf["kl_cap"] > 1.2 * pf["q_KLT"], "the KL cap binds inside the range the memo quotes"),
        (pf["q_N"] <= N_MAX, "the matched best-of-n budget is outside the build's range"),
        (0.05 < pf["q_KLT"] <= KL_MAX, "the proposed KL target is outside the knob's range"),
        (10 <= pf["q_ST"] <= 2000, "the proposed step count is outside the knob's range"),
        (10 <= pf["q_Slight"] <= 2000, "the light PPO option is outside the knob's range"),
        (0.0 <= pf["q_lpB"] <= LP_MAX, "the q4 length penalty is outside the knob's range"),
        (4 <= pf["q_nA"] <= N_MAX and 4 <= pf["q_nB"] <= N_MAX, "a q4 best-of-n option is out of range"),
        (0.2 <= _q2_truth(pf) <= 0.8, "the length share is too close to 0 or 1 to be interesting"),
        (pf["q_nmemo"] >= 2.5 * pf["q_N"], "the memo's dismissal of best-of-n is not quantitatively wrong"),
    ]:
        if not ok:
            return False, why
    return True, "admissible"


def wellposed(w):
    """Admissibility of the world, of the evidence drawn in it, and of the world *set* q3 is graded in.

    The last check is the one that matters.  `plan_gate` re-runs every declared world against the disclosed
    notebook at the grading z-bar and refuses the instance if any of them is ruled out; a notebook that its
    own world happens to sit 2.5 sigma from would fail that for reasons with nothing to do with the plan."""
    from .. import verify as V
    ok, why = _admissible(w["pf"])
    if not ok:
        return False, why
    pf = R.full(w["pf"])
    c, det = V.consistency(R, pf, w["rows"], z_bar=ZB)
    if not c or det["max_abs_z"] > 0.75 * ZB or det["sum_z2"] > 0.80 * det["chi2_bar"]:
        return False, ("the drawn notebook sits too far from its own world: max |z| %.2f, sum z^2 %.1f of "
                       "%.1f" % (det["max_abs_z"], det["sum_z2"], det["chi2_bar"]))
    # The q2 estimator extrapolates a saturating length term back to zero distance, and how well it does
    # that is a property of the world, not of the code: when the sweep's far end still carries a large
    # length term the extrapolation is long and the dispersion is wide.  Measure it, and refuse any world
    # where the tolerance that dispersion would earn could reach the nearest wrong answer.  Without this
    # screen the calibration silently trades `skip:baseline` and `skip:extrapolate` away for a pass.
    from .. import lab as L
    share = _q2_truth(w["pf"])
    seps = [share, 1.0 - share,
            abs(w["pf"]["q_vbase"] - share), abs(w["pf"]["q_vstop"] - share)]
    errs = []
    for t in range(Q2_MC):
        sess = L.Session(w["pf"], w["spec"], "%s/mc%d" % (w["salt"], t))
        sess.caps = {"run_cost": 1e12, "total_cost": 1e15, "max_runs": 10 ** 6}
        est, _ = _q2_measure(sess, np.random.default_rng(90000 + 7919 * t))
        errs.append(abs(est - share))
    p90 = float(np.percentile(errs, 90))
    need = 2.0 * max(FLOOR_SHARE, 2.25 * p90) * Q2_MC_SLACK
    if need > min(seps):
        return False, ("the q2 estimator is too disperse in this world to separate its rivals: p90 error "
                       "%.4f wants %.3f of room, nearest wrong answer is %.3f away" % (p90, need, min(seps)))

    worlds = _q3_worlds(w["p"])
    for wd in worlds:
        cw, _ = V.consistency(R, R.full(wd["params"]), w["rows"], z_bar=ZB)
        if not cw:
            return False, "world %s is not consistent with the disclosed notebook" % wd["name"]
    labs = [wd["label"] for wd in worlds]
    if sorted(labs) != ["bon", "ppo", "undetermined"]:
        return False, "the three consistent worlds do not license three different conclusions: %s" % labs
    return True, (why + "; notebook max |z| %.2f, sum z^2 %.1f of %.1f; worlds %s"
                  % (det["max_abs_z"], det["sum_z2"], det["chi2_bar"], "/".join(labs)))
