"""E02 speculative tail: an inference team has to decide whether to ship a draft scheduler that raises the
proposal length from 4 to 8, and what one production replica will actually deliver at its capacity ceiling.

(O11 extrapolation past the measured region, O4 measurement artifact, O2 confounded attribution,
O13 regime change.)

The build installed in the lab caps `spec_g` at 4, so every acceptance rate the team can measure lives at
positions 0-3.  The claim they want to make is about positions 4-7.  S3's decay beyond `tail_from` is a
*separate* constant (`rho_tail`), it is a documented known unknown, and nothing runnable touches it - so
"does proposal length 8 beat 4" is not a measurement, it is a question about which continuations the
evidence leaves open.  That is what the two counterexample questions ask, from the two sides: one claim is
true in this world but *not entailed* (a slower tail reproduces every number the team has and reverses it),
the other is a bound that **is** entailed, and the analyst who says "unmeasurable, therefore unbounded" is
wrong.  The witness for the first has to move the acceptance constants too - moving `rho_tail` alone leaves
the measured positions where they are and the evidence still points the other way - so it lives in the thin
diagonal band the profile fit leaves open, a few per cent of the declared box.

q1 is the confound.  Every aggregate the memo quotes mixes two things: how much *yield* speculation buys
(the acceptance profile) and what it *costs* (the draft model's time per proposed token).  The memo
attributes the whole of the observed speedup to acceptance.  Separating them needs a matched baseline (the
memo's is at a different context and batch, where the decode step itself is a different length), the
per-position profile rather than the reported mean `accept_rate`, and the observation that `ms_per_token`
is per *accepted output token*, not per verification step - so the step-time ratio is `(ms4/ms0) * E4`.
Skip that last step and the arithmetic hands back a *negative* drafting cost; noticing the contradiction is
the point.  The published c = 0.128 (Leviathan 2023) is not this lab's constant.

q2 is the capacity question and the one place S1 and S5 interact.  The production replica reserves a fixed
KV pool - smaller than the free memory on the team's dev box, exactly as a serving stack with a
`gpu_memory_utilization` setting does - and fills it with whole sequences of the p95 context length.  That
length is undecided (the second known unknown), so the answer is a set; at the short end the replica is
past the critical batch and sits on the compute roof, at the long end it is memory-bound and well under it.
Three shortcuts each collapse it: reading the lab box's own `batch_max` instead of the production pool,
applying the memory-traffic formula everywhere and missing the roof, and answering at the midpoint as if
the range did not matter.

Useful designs: a matched `spec_g=0` / `spec_g=4` pair at one `(seq, batch)` - the decode step cancels out
of the ratio only if both sides are at the same configuration; long `dur` on those rows, since the
acceptance positions are estimated from the proposals the run had time to make; `bench` at the two
production points directly, which is possible precisely because the reserved pool is smaller than the free
memory the lab reports; and `batch_max` / `kv_bytes_per_seq`, which are exact and cost whatever the
shortest run costs.
"""
import json, math
import numpy as np
from .. import queries as Q
from ..common import draw_card, run_rows
from ..labs import servelab as SV

ID = "e02-spec-tail"
TITLE = "Speculative tail: what a proposal length you cannot run would buy, and what one replica delivers"
CARDS = ["S1", "S3", "S5"]
OBSTACLES = ["O11", "O4", "O2", "O13"]
CLAIM = ("separate a speculative decoder's yield from its cost, price a proposal length the build cannot "
         "run as a question about what the evidence leaves open rather than as a measurement, and read a "
         "production replica's capacity off the pool it actually reserves")
DIFFICULTY = {"depth": 3, "nuisance": ["S1", "S3", "S5"], "anti_prior": ["q1"]}
EXEMPT_LOAD_BEARING = {}
CERT_N = 400

DRAFT = "d_lite"
BITS = 16
TAIL_FROM = 4                    # positions 0-3 decay with `rho`; 4 and beyond with `rho_tail`
G_MAX = 4                        # the shipped build's cap on the proposal length: the whole point
G_ASK = 8                        # the proposal length the new scheduler would use
RT_LO, RT_HI = 0.85, 1.00        # S3's documented range for the tail decay (the known unknown)
C_PUB = 0.128                    # Leviathan et al. 2023: the published draft cost ratio
NREP_NB, DUR_NB = 3, 40.0        # notebook: gamma=4 replicates and their duration
DUR_CAP = 20.0                   # notebook / oracle duration on the two capacity rows
R_Q1, R_Q2 = 5, 4                # oracle replicates: matched-pair rows, capacity rows
BOX_W, BOX_REL = 8.0, 0.25       # witness box: BOX_W half-widths wide, truth at relative position BOX_REL
# The z-radius the two counterexample items are graded at, declared on the items themselves and used
# everywhere the consistent set is computed.  The package default of 3 is wrong for this task and would
# make it unanswerable: `consistency` takes the maximum |z| over *every* observable in the evidence, and
# this lab reports six per speculative row, so an agent that spends its 36 runs accumulates about 216 of
# them.  Under the default even the *true* world - and therefore every witness, since a witness differs
# from it only in a constant nothing measures - fails with probability 1 - 0.9973^216 = 44%: the harder
# the agent works, the likelier its correct answer is rejected.  At 4.5 that probability is 0.15%, and the
# price is small here because the consistent set is a thin diagonal band along which the quantity q4
# bounds barely varies: widening the band by half moves the sup by 0.2%.
ZB_GRADE = 4.5
A_BOX, RHO_BOX, C_BOX = (0.884, 0.920), (0.930, 0.950), (0.050, 0.076)
W_MIN, W_MAX = 1.22, 1.60        # q2 set width, as khi/klo
ROOF_MIN, EX_MIN, CAP_MIN = 1.12, 0.09, 1.15
GAP1_MIN, PRIOR_MIN = 0.040, 0.10
CONT_MIN, ROOM_MIN, REV_MAX = 1.012, 1.040, 0.990
RTS_LO, RTS_HI, RTS_PAD = 0.875, 0.975, 0.012
VOL_LO, VOL_HI = 0.006, 0.075    # proxy screen at draw time
RVOL_LO, RVOL_HI = 0.008, 0.085  # the real criterion, checked once in wellposed
MATCH_MIN = 0.15                 # |step(benchmark config) / step(capacity probe) - 1|


def _set(p, path, v):
    node = p
    parts = path.split(".")
    for s in parts[:-1]:
        node = node[s]
    node[parts[-1]] = v
    return p


# --------------------------------------------------------------------------- S3 algebra (noise-free)
def _dp(a, rho, c, rt):
    return {"drafts": {DRAFT: {"a": a, "rho": rho, "c": c, "tail_from": TAIL_FROM, "rho_tail": rt}}}


def _prof(a, rho, g=G_MAX):
    return SV.accept_profile(_dp(a, rho, 0.0, rho), DRAFT, g)


def _E(a, rho, g, rt=None):
    """Expected accepted tokens per verification step, through the backend's own law."""
    return SV.exp_tokens(_dp(a, rho, 0.0, rho if rt is None else rt), DRAFT, g)


def _ratio(a, rho, c, rt):
    """tokens/s at proposal length G_ASK divided by tokens/s at G_MAX, at any fixed (seq, batch).

    The decode step is common to both and cancels, so this is a property of the draft alone."""
    return (_E(a, rho, G_ASK, rt) / _E(a, rho, G_MAX, rt)) * (1.0 + c * G_MAX) / (1.0 + c * G_ASK)


def _flat_a(a, rho):
    """What `accept_rate` reports at gamma=G_MAX: the mean of the measured positions."""
    return sum(_prof(a, rho)) / float(G_MAX)


def _E_flat(af, g):
    e, pre = 1.0, 1.0
    for _ in range(g):
        pre *= af
        e += pre
    return e


def _loose_tail(a, rho):
    """The extra expected tokens positions G_MAX..G_ASK-1 contribute when the tail is bounded only by
    `a_i <= a_0` - i.e. by nothing except acceptance never exceeding its value at the first position."""
    pre = 1.0
    for x in _prof(a, rho):
        pre *= x
    out = 0.0
    for _ in range(G_ASK - G_MAX):
        pre *= a
        out += pre
    return out


def _bisect(f, lo, hi, n=60):
    flo = f(lo) > 0
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == flo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _rts(a, rho, c):
    """The tail decay at which G_ASK exactly ties G_MAX.  Below it the claim of q3 is false."""
    f = lambda rt: _ratio(a, rho, c, rt) - 1.0
    return _bisect(f, 0.30, 1.0) if f(0.30) < 0 else 0.0


# ------------------------------------------------- the consistency geometry the witness box sits in
def _zmax(a, rho, a0, r0, nrep=NREP_NB, dur=DUR_NB):
    """max |z| of world (a, rho) against the *ideal* observables of `nrep` gamma=G_MAX rows at (a0, r0).

    This is the build-time proxy for `verify.consistency`: same observables (the acceptance positions,
    `tokens_per_s`, `ms_per_token`), same standard deviations, with the noise-free predictions standing in
    for the draw.  It ignores the chi-square half of the real criterion, which only ever *shrinks* the
    consistent set - so a sup certified here is conservative, which is the direction that matters.  The
    one place the difference could bite the other way, the size of the witness set, is measured against
    the real criterion in `wellposed` rather than trusted to this."""
    d = dur / SV.DUR_REF
    sig_t, sig_l = SV.BASE["sig_tput"] / math.sqrt(d), SV.BASE["sig_lat"] / math.sqrt(d)
    nacc = SV.N_ACC * d
    zs = []
    for o, m in zip(_prof(a0, r0), _prof(a, rho)):
        sd = math.sqrt(max(m * (1 - m), 1e-9) / nacc) / math.sqrt(nrep)
        zs.append((o - m) / sd)
    Et, Ec = _E(a0, r0, G_MAX), _E(a, rho, G_MAX)
    zs.append((Et / Ec - 1.0) / (sig_t / math.sqrt(nrep)))     # tokens_per_s is proportional to E
    zs.append((Ec / Et - 1.0) / (sig_l / math.sqrt(nrep)))     # ms_per_token to 1/E
    return max(abs(z) for z in zs)


def _half(a0, r0, key, lo=0.0, hi=0.15):
    """How far one constant can move on its own before a three-sigma reading rejects it.

    This is the unit the witness box is measured in, and it stays at 3 rather than following ZB_GRADE:
    the box only has to be wide enough to be a real question and narrow enough that its corners are
    rejected outright, and pinning it to a fixed multiple of the notebook's own resolution keeps it from
    moving when the grading radius is retuned."""
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        a, r = (a0 + mid, r0) if key == "a" else (a0, r0 + mid)
        if _zmax(a, r, a0, r0) > 3.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _box(a0, r0):
    """The declared free-parameter ranges of the two counterexample questions.

    Each is BOX_W half-widths wide with the truth at relative position BOX_REL, so every corner of the box
    - which is where an agent guesses first, and which G14 checks explicitly - sits at least two
    half-widths (about six sigma) from the truth on `a` or on `rho`, and the evidence rejects it outright.
    The width is not a secret: the profile is measurable to a few thousandths from the notebook alone, so
    the box tells the agent nothing it could not fit in one line.  What it does is make the question
    well-posed - a witness is an assignment inside *this* box - and stop "refutable" from being winnable by
    naming an absurd world."""
    ha, hr = _half(a0, r0, "a"), _half(a0, r0, "rho")
    alo = a0 - BOX_REL * BOX_W * ha
    rlo = r0 - BOX_REL * BOX_W * hr
    return {"a": (round(alo, 4), round(alo + BOX_W * ha, 4)),
            "rho": (round(rlo, 4), round(min(rlo + BOX_W * hr, 0.9995), 4)),
            "rho_tail": (RT_LO, RT_HI), "ha": ha, "hr": hr}


def _sup_z(a0, r0, c, zb=ZB_GRADE, n=61, rt=RT_HI):
    """sup of the G_MAX -> G_ASK throughput ratio over the worlds the evidence leaves open, at the most
    favourable tail.  Scanned on a grid of the |z| <= zb region rather than solved: the region is a thin
    curved band (raising `a` and lowering `rho` keeps the measured positions where they are), and the
    bounding rectangle overstates the sup by enough to make the entailed bound of q4 unkillable.  The
    maximiser lands near (0.9 ha, 0.8 hr) and is interior, so the grid width is not what sets it."""
    ha, hr = _half(a0, r0, "a"), _half(a0, r0, "rho")
    f = zb / 3.0
    best = -9.0
    for i in range(n):
        a = a0 + 1.6 * f * ha * (2.0 * i / (n - 1.0) - 1.0)
        if a >= 1.0:
            continue
        for j in range(n):
            r = r0 + 1.6 * f * hr * (2.0 * j / (n - 1.0) - 1.0)
            if r >= 1.0 or _zmax(a, r, a0, r0) > zb:
                continue
            best = max(best, _ratio(a, r, c, rt))
    return best


def _vol_proxy(a0, r0, c, box, n=3000, seed=7):
    """Cheap estimate of G14's witness volume for q3: the share of the declared box that reproduces the
    evidence (by the proxy above) and reverses the claim.  A screen, not the certificate."""
    rng = np.random.default_rng(seed)
    (alo, ahi), (rlo, rhi) = box["a"], box["rho"]
    nb = 0
    for _ in range(n):
        a, r = float(rng.uniform(alo, ahi)), float(rng.uniform(rlo, rhi))
        rt = float(rng.uniform(RT_LO, RT_HI))
        if a < 1.0 and r < 1.0 and _zmax(a, r, a0, r0) <= ZB_GRADE and _ratio(a, r, c, rt) <= 1.0:
            nb += 1
    return nb / float(n)


def _bound(a, rho, c):
    """The bound q4 asserts, placed between what the evidence actually allows and what the two standard
    over-estimates allow.  The geometric mean of the two is used rather than a midpoint so the margin is
    proportional on both sides: the analyst who models the profile as flat (the mean `accept_rate` the
    lab reports) and the analyst who bounds the tail only by `a_i <= a_0` both overshoot it, and the sup
    over consistent worlds sits ROOM_MIN below it.  Returned as a *fractional* increase, which is what the
    question prints and what `claim_fn` compares against."""
    sup = _sup_z(a, rho, c)
    af = _flat_a(a, rho)
    flat = (_E_flat(af, G_ASK) / _E_flat(af, G_MAX)) * (1.0 + c * G_MAX) / (1.0 + c * G_ASK)
    E4 = _E(a, rho, G_MAX)
    loose = ((E4 + _loose_tail(a, rho)) / E4) * (1.0 + c * G_MAX) / (1.0 + c * G_ASK)
    return math.sqrt(sup * min(flat, loose)) - 1.0, sup, flat, loose


# ----------------------------------------------------------------- S1 / S5 capacity (noise-free)
def _plateau(pf):
    return pf["P_peak"] * pf["eff"] / (2.0 * pf["N"])


def _tp(pf, seq, batch):
    """Steady-state decode throughput without speculation: one token per step, `batch` in flight."""
    return batch / SV.step_time(pf, BITS, seq, batch)


def _pool_batch(pf, seq):
    """Whole sequences of `seq` tokens that fit in the replica's reserved KV pool."""
    return int(pf["kv_pool"] // (SV.kv_per_token(pf) * float(seq)))


def _cap(pf):
    """Every capacity number the items and the rivals are built from, noise-free."""
    slo, shi = int(pf["ctx_lo"]), int(pf["ctx_hi"])
    mid = (slo + shi) // 2
    bl, bm, bh = _pool_batch(pf, shi), _pool_batch(pf, mid), _pool_batch(pf, slo)
    out = {"slo": slo, "shi": shi, "mid": mid, "bl": bl, "bm": bm, "bh": bh, "plateau": _plateau(pf),
           "bmax_lo": SV.batch_max(pf, BITS, slo), "bmax_hi": SV.batch_max(pf, BITS, shi)}
    out["khi"] = _tp(pf, slo, bh) if bh > 0 else 0.0                  # short context -> on the roof
    out["klo"] = _tp(pf, shi, bl) if bl > 0 else 0.0
    out["kmid"] = _tp(pf, mid, bm) if bm > 0 else 0.0
    # the three shortcuts, as numbers
    out["box_lo"] = min(_tp(pf, shi, out["bmax_hi"]) if out["bmax_hi"] > 0 else 0.0, out["plateau"])
    mem = lambda s, b: b * pf["BW"] / (SV.weight_bytes(pf, BITS) + b * SV.kv_per_token(pf) * float(s))
    out["mem_hi"] = mem(slo, bh) if bh > 0 else 0.0                   # no compute roof
    return out


# ------------------------------------------------------------------------------------------ world
def draw(rng):
    why = "no draw attempted"
    for _ in range(800):
        p, why = _draw_hw(rng)
        if p is None:
            continue
        sp, why = _draw_spec(rng)
        if sp is None:
            continue
        p["drafts"] = {"none": {"a": 0.0, "c": 0.0, "rho": 1.0},
                       DRAFT: {"a": sp["a"], "rho": sp["rho"], "c": sp["c"],
                               "tail_from": TAIL_FROM, "rho_tail": sp["rt_true"]}}
        # the bound of q4 is fixed at draw time and printed in the question, so it must NOT be recomputed
        # inside a witness world: a witness moves `a` and `rho`, and a bound that moved with them would
        # make the claim unfalsifiable by construction.  Stashing it in the params is enough - `full()`
        # copies BASE and then updates, so blueprint-private keys survive into the session.
        p["x4"] = sp["x4"]
        pf = SV.full(p)
        cap = _cap(pf)
        # the configuration the spec build was benchmarked at: deliberately not one of the production
        # points, so that the memo's cross-configuration comparison of `ms_per_token` is wrong
        ok = False
        for _try in range(40):
            seq_b = int(rng.integers(4, 13)) * 128
            bmx = SV.batch_max(pf, BITS, seq_b)
            if bmx < 24:
                continue
            batch_b = int(rng.integers(16, max(17, min(bmx, 4 * max(cap["bl"], 4)) + 1)))
            t_b = SV.step_time(pf, BITS, seq_b, batch_b)
            t_p = SV.step_time(pf, BITS, cap["shi"], cap["bl"])
            if abs(t_b / t_p - 1.0) >= MATCH_MIN:
                ok = True
                break
        if not ok:
            why = "no benchmarking configuration far enough from the capacity probe"
            continue
        p["bench_seq"], p["bench_batch"] = seq_b, batch_b
        # scenario unknown: the production p95 context length.  Nothing in the lab reads it.
        p["prod_seq"] = float(rng.uniform(cap["slo"], cap["shi"]))
        ok, why = _admissible(SV.full(p))
        if ok:
            return p
    raise RuntimeError("no admissible draw (last: %s)" % why)


def _draw_hw(rng):
    """Hardware, model size and the production deployment, drawn *constructively*.

    Blind rejection does not work here.  Three identities tie the answer together: with `r = ctx_hi/ctx_lo`
    and `w = khi/klo`, the factor by which the memory-traffic formula overstates the short-context endpoint
    is exactly `r / w`; a midpoint answer's shortfall is `min(w(r+1)/(2r) - 1, (r-1)/(2r))`; and the
    reserved pool that produces a given `w` is `pool = x W / (w - x)` with `x = kv_per_token * ctx_hi *
    P_peak * eff / (2 N BW)`.  So `r` is drawn, the window of `w` that satisfies both the roofline and the
    midpoint margins is computed from it, `w` is drawn inside that window, and the pool identity is
    inverted into a window on `ctx_hi`.  Drawing the same quantities freely and testing afterwards accepted
    0.14% of draws; this accepts about two thirds, and the post-screen in `_admissible` still has the last
    word because the batch floors are integer."""
    N = float(np.exp(rng.uniform(math.log(8.0e9), math.log(2.2e10))))
    p = dict(N=N, L=int(rng.integers(28, 65)), H_kv=8, d_head=128, M_gpu=80.0e9,
             act_bytes=float(rng.uniform(1.4e9, 2.8e9)),
             P_peak=float(rng.uniform(3.4e14, 4.6e14)), BW=float(rng.uniform(2.8e12, 4.2e12)))
    # `draw_card` iterates over the card's own parameter list, so only genuine card parameters go through
    # it: S1 declares `eff`, S5 declares `mem_util` and `kv_bytes`.  Anything else named here would be
    # silently dropped and the world would quietly fall back to the backend defaults.
    p.update(draw_card(rng, "S1", {"eff": (0.50, 0.74)}))
    p.update(draw_card(rng, "S5", {"mem_util": (0.84, 0.92)}))
    p.pop("prefill_flop_eff", None)                      # S1's other parameter: no prefill in this task
    pf = SV.full(p)
    W = SV.weight_bytes(pf, BITS)
    u = SV.kv_per_token(pf)
    A = 2.0 * N * pf["BW"] / (pf["P_peak"] * pf["eff"])       # KV bytes of context at the critical batch
    free = pf["M_gpu"] * pf["mem_util"] - W - pf["act_bytes"]
    if free < 6.0e9:
        return None, "only %.1f GB free on the dev box" % (free / 1e9)
    r = float(rng.uniform(1.45, 2.20))
    wlo = max(W_MIN, (1.0 + EX_MIN) * 2.0 * r / (r + 1.0))
    whi = min(W_MAX, r / ROOF_MIN)
    if wlo >= whi:
        return None, "no width window at r = %.2f" % r
    w = float(rng.uniform(wlo, whi))
    g1, g2 = 0.15 * free / W, 0.90 * free / W                 # the pool is 15-90% of what is free
    xlo, xhi = w * g1 / (1.0 + g1), w * g2 / (1.0 + g2)
    lo = max(xlo * A / u, 1024.0, 640.0 * r)
    hi = min(xhi * A / u, 4096.0)
    if lo >= hi:
        return None, "no context window at r = %.2f, w = %.2f" % (r, w)
    shi = int(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    x = u * shi / A
    p["ctx_lo"], p["ctx_hi"] = int(shi / r), shi
    p["kv_pool"] = float(x * W / (w - x))
    return p, "ok"


def _draw_spec(rng):
    """The draft model's acceptance profile and cost.  Rejection is fine here - the feasible region is a
    solid block, unlike the hardware one - and A_BOX / RHO_BOX / C_BOX are that block, found by a grid
    scan over the S3 card's own ranges, so roughly a fifth of draws land.

    The card's own `rho_tail` draw is discarded: this task needs the tail placed relative to the tie point,
    which is a function of the other three constants and is not known until they are drawn."""
    d = draw_card(rng, "S3", {"a0": A_BOX, "rho": RHO_BOX, "c": C_BOX})
    a, rho, c = float(d["a0"]), float(d["rho"]), float(d["c"])
    rts = _rts(a, rho, c)
    if not RTS_LO <= rts <= min(RTS_HI, rho - RTS_PAD):
        return None, "the tie point rho_tail = %.4f is outside the window" % rts
    x4, sup, flat, loose = _bound(a, rho, c)
    if min(flat, loose) / sup < ROOM_MIN:
        return None, "no room between sup %.4f and the naive bounds %.4f" % (sup, min(flat, loose))
    # the true tail sits above the tie point, so the claim of q3 is true in this world, and below the top
    # of the documented range, so the analyst who assumes the decay simply continues is wrong as well
    rt_true = float(rng.uniform(max(rts + 0.012, 0.5 * (rts + RT_HI)), RT_HI - 0.005))
    return {"a": a, "rho": rho, "c": c, "rt_true": rt_true, "x4": round(x4, 4)}, "ok"


def spec(p):
    return {"lab": "servelab",
            "knobs": {"svc": {"type": "choice", "values": ["bench"], "default": "bench"},
                      "seq": {"type": "float", "min": 256, "max": 4096, "int": True, "default": 1024},
                      "batch": {"type": "float", "min": 1, "max": 1024, "int": True, "default": 32},
                      "draft": {"type": "choice", "values": ["none", DRAFT], "default": "none"},
                      "spec_g": {"type": "float", "min": 0, "max": G_MAX, "int": True, "default": 0},
                      "dur": {"type": "float", "min": 5.0, "max": 40.0, "default": 20.0}},
            "fixed": {"bits": BITS},
            # `dur` is the precision knob and the binding constraint: the matched pair that separates the
            # draft's cost from its yield wants long runs on both sides, and the capacity question wants
            # two more configurations.  The reference solution uses 18 runs and 560 of the 900.
            "caps": {"run_cost": 40.0, "total_cost": 900.0, "max_runs": 36}}


def LAB_EXTRA(w):
    pf = w["pf"]
    return ["",
            "This lab serves one model at `bits=16` and exposes the decode benchmark only.  The build "
            "installed here caps the proposal length at `spec_g=%d`; the scheduler that would raise it is "
            "not in this build and cannot be run." % G_MAX,
            "",
            "`draft=%s` is the draft model the team benchmarked.  A `bench` row with `spec_g>0` reports "
            "`accept_by_position` (the acceptance rate estimated at each proposal position, in order) and "
            "`accept_rate` (their mean).  `ms_per_token` is wall time per **accepted output token**, not "
            "per verification step." % DRAFT,
            "",
            "`kv_bytes_per_seq` and `batch_max` are exact and describe **this machine**: `batch_max` is the "
            "largest batch whose weights, activations and KV cache fit in its memory.  The production "
            "deployment in `/app/deploy/serving.yaml` reserves a fixed KV pool of its own, which is a "
            "different and smaller number (%.2f GB)." % (pf["kv_pool"] / 1e9)]


def known_unknowns(p):
    pf = SV.full(p)
    return [
        {"name": "production p95 context length", "param": "prod_seq",
         "range": [float(pf["ctx_lo"]), float(pf["ctx_hi"])],
         "text": "The production replicas serve a traffic mix whose p95 context length lies between %d and "
                 "%d tokens; capacity planning uses that p95 length, and the traffic team has not fixed a "
                 "single figure.  Nothing in this service depends on the choice.  A question about the "
                 "production replica must cover the whole range."
                 % (int(pf["ctx_lo"]), int(pf["ctx_hi"]))},
        {"name": "acceptance decay beyond proposal position %d" % TAIL_FROM,
         "param": "drafts.%s.rho_tail" % DRAFT, "range": [RT_LO, RT_HI],
         "text": "The draft model's per-position acceptance decays geometrically, and the decay factor "
                 "from proposal position %d on is a *separate* constant from the one governing positions "
                 "0 to %d.  It is known only to lie in [%.2f, %.2f].  No configuration this build can run "
                 "exercises a position past %d, so this service behaves identically for every value in "
                 "that range."
                 % (TAIL_FROM, TAIL_FROM - 1, RT_LO, RT_HI, TAIL_FROM - 1)}]


# --------------------------------------------------------------------------------------- notebook
def _nb_reqs(pf):
    cap = _cap(pf)
    reqs = [{"seq": int(pf["bench_seq"]), "batch": int(pf["bench_batch"]), "draft": DRAFT,
             "spec_g": G_MAX, "dur": DUR_NB, "seed": 100 + k} for k in range(NREP_NB)]
    reqs.append({"seq": cap["shi"], "batch": cap["bl"], "draft": "none", "spec_g": 0,
                 "dur": DUR_CAP, "seed": 200})
    reqs.append({"seq": cap["slo"], "batch": cap["bh"], "draft": "none", "spec_g": 0,
                 "dur": DUR_CAP, "seed": 201})
    return reqs


def notebook(p, sess, rng):
    pf = SV.full(p)
    cap = _cap(pf)
    rows = []
    for r in _nb_reqs(pf):
        cfg, seed, ex = sess.validate(dict(r))
        rows.append(sess.execute(cfg, seed, ex))
    spec_rows, probe_hi, probe_lo = rows[:NREP_NB], rows[NREP_NB], rows[NREP_NB + 1]
    ms_spec = float(np.mean([r["ms_per_token"] for r in spec_rows]))
    tp_spec = float(np.mean([r["tokens_per_s"] for r in spec_rows]))
    acc = float(np.mean([r["accept_rate"] for r in spec_rows]))
    prof = [float(np.mean([r["accept_by_position"][i] for r in spec_rows])) for i in range(G_MAX)]
    speed = probe_hi["ms_per_token"] / ms_spec
    e8f, e4f = _E_flat(acc, G_ASK), _E_flat(acc, G_MAX)
    c_share = 100.0 * G_MAX * C_PUB / (1.0 + G_MAX * C_PUB)
    notes = """# Draft-scheduler review (inference team)

We benchmarked the speculative build against our previous numbers and we are ready to recommend the
longer-proposal scheduler.  Runs are in `notebook/runs.jsonl`; the deployment we are sizing for is in
`deploy/serving.yaml`.

- **Speculation is working.**  At `seq=%d batch=%d` with `draft=%s spec_g=%d` we measure `ms_per_token`
  %.4f and `tokens_per_s` %.1f (runs 1-3, three repetitions at `dur=%.0f`).  Our non-speculative reference
  point is run 4, `ms_per_token` %.4f, so the build is **%.2fx** faster per output token.
- **Acceptance.**  `accept_rate` averages %.4f over the three runs (per position: %s).  We are treating
  %.4f as *the* acceptance rate of this draft model.
- **Cost of drafting.**  The draft is about a tenth of the target and the published figure for a draft of
  that ratio is c = %.3f, i.e. drafting adds %.1f%% per proposed token.  At `spec_g=%d` that is %.1f%% of
  the step, which the %.2fx above more than pays for.  We have not measured c ourselves - the speedup is
  what matters.
- **Longer proposals.**  Acceptance is still %.4f at position %d, so there is clearly headroom.  Holding it
  flat at the %.4f average through positions %d-%d gives an expected %.2f accepted tokens per step at
  `spec_g=%d` against %.2f at `spec_g=%d`; net of the extra drafting that is a further gain.
  **Recommendation: ship the `spec_g=%d` scheduler.**
- **Capacity.**  This box reports `batch_max` %d at %d tokens of context and %d at %d (runs 4-5), so one
  replica handles up to %d concurrent sequences at the long end of our context mix.  Sizing off the short
  end (run 5: %.1f tok/s at batch %d) we expect roughly %.0f tok/s per replica, and the capacity plan will
  quote the midpoint of the context range.

Open questions nobody has answered: our two reference points sit at different contexts and batches, and
the decay past position %d is not something this build can measure.
""" % (int(pf["bench_seq"]), int(pf["bench_batch"]), DRAFT, G_MAX,
       ms_spec, tp_spec, DUR_NB,
       probe_hi["ms_per_token"], speed,
       acc, "/".join("%.4f" % v for v in prof), acc,
       C_PUB, 100.0 * C_PUB, G_MAX, c_share, speed,
       prof[-1], G_MAX - 1, acc, G_MAX, G_ASK - 1, e8f, G_ASK, e4f, G_MAX, G_ASK,
       cap["bmax_hi"], cap["shi"], cap["bmax_lo"], cap["slo"], cap["bmax_hi"],
       probe_lo["tokens_per_s"], cap["bh"], cap["kmid"],
       G_MAX - 1)
    box = _box(pf["drafts"][DRAFT]["a"], pf["drafts"][DRAFT]["rho"])
    ctx = {"rows": rows, "cap": cap, "box": box, "ms_spec": ms_spec, "tp_spec": tp_spec, "acc": acc,
           "prof": prof, "speed": speed, "ms_probe": probe_hi["ms_per_token"],
           "tp_probe": probe_lo["tokens_per_s"], "e8f": e8f, "e4f": e4f, "c_share": c_share,
           "obs": [v for r in rows for v in (r["tokens_per_s"], r["ms_per_token"],
                                             r["kv_bytes_per_seq"], r["batch_max"], r["batch"])],
           "acc_obs": [v for r in spec_rows for v in (list(r["accept_by_position"]) + [r["accept_rate"]])],
           "pub": [pf["bench_seq"], pf["bench_batch"], pf["kv_pool"], pf["kv_pool"] / 1e9,
                   pf["ctx_lo"], pf["ctx_hi"], pf["x4"], 100.0 * pf["x4"],
                   C_PUB, 100.0 * C_PUB, RT_LO, RT_HI, G_MAX, G_ASK, TAIL_FROM, DUR_NB, DUR_CAP,
                   box["a"][0], box["a"][1], box["rho"][0], box["rho"][1]]}
    return rows, notes, ctx


def files(p, ctx, rng):
    cap = ctx["cap"]
    yaml = """# production serving configuration (replica template) -- do not edit by hand
model:
  name: target-16b
  dtype: float16
runtime:
  # bytes of device memory reserved for the paged KV cache on every replica.  Fixed at deploy time, and
  # smaller than the free memory on a development box so that the profiler and the tracer have room.
  kv_pool_bytes: %d
  block_size: 16
  enforce_whole_sequences: true      # a sequence is admitted only if its full context fits
  speculative:
    draft: %s
    num_speculative_tokens: %d       # the shipped build; %d needs the new scheduler
traffic:
  # the p95 context length of the production mix, as measured by the traffic team
  ctx_p95_min_tokens: %d
  ctx_p95_max_tokens: %d
""" % (int(p["kv_pool"]), DRAFT, G_MAX, G_ASK, cap["slo"], cap["shi"])
    return {"notebook/runs.jsonl": "\n".join(json.dumps(r, sort_keys=True) for r in ctx["rows"]) + "\n",
            "deploy/serving.yaml": yaml}


# ------------------------------------------------------------------------------------------ items
def _keys(pf):
    c = pf["drafts"][DRAFT]["c"]
    cap = _cap(pf)
    return {"f": G_MAX * c / (1.0 + G_MAX * c), "lo": cap["klo"], "hi": cap["khi"]}


def _schema(box):
    mk = lambda name, path: {"path": ["drafts", DRAFT, path], "lo": box[name][0], "hi": box[name][1],
                             "type": "float"}
    return {"params": {"a": mk("a", "a"), "rho": mk("rho", "rho"), "rho_tail": mk("rho_tail", "rho_tail")}}


def claim_fn(qid, p):
    """Both counterexample questions are about the same ratio, read out of whatever world is handed in.

    `x4` is read, never recomputed: it was fixed at draw time and printed in the question."""
    d = p["drafts"][DRAFT]
    rt = d["rho"] if d.get("rho_tail") is None else d["rho_tail"]
    r = _ratio(float(d["a"]), float(d["rho"]), float(d["c"]), float(rt))
    return r > 1.0 if qid == "q3" else r <= 1.0 + float(p["x4"])


def items(p, ctx, tol=None):
    tol = tol or {}
    key = _keys(p)
    box = _box(p["drafts"][DRAFT]["a"], p["drafts"][DRAFT]["rho"])
    sch = _schema(box)
    fp = G_MAX * C_PUB / (1.0 + G_MAX * C_PUB)
    ranges = ("`a` (acceptance at proposal position 0) in [%.4f, %.4f], `rho` (the decay over positions 0 "
              "to %d) in [%.4f, %.4f], and `rho_tail` (the decay from position %d on) in [%.2f, %.2f].  A "
              "witness gives all three and nothing else."
              % (box["a"][0], box["a"][1], G_MAX - 1, box["rho"][0], box["rho"][1],
                 TAIL_FROM, RT_LO, RT_HI))
    return [
        Q.point("q1", "With `draft=%s` at proposal length %d: the fraction of one verification step's wall "
                      "time that goes to the draft model rather than to the target model's verification "
                      "pass.  A fraction in [0, 1]; it is the same at every batch and context length "
                      "(noise-free expected value)." % (DRAFT, G_MAX),
                "fraction", key["f"], tol.get("q1", 0.02), floor=0.004,
                cards=["S3"], obstacles=["O2", "O4"], chain=["matched", "profile", "perstep"],
                prior_key={"lo": fp, "hi": fp}),
        Q.interval("q2", "Steady-state decode throughput, in output tokens per second, of **one production "
                         "replica** under `deploy/serving.yaml`: no speculation, `bits=16`, the replica "
                         "filled with as many whole sequences of the p95 context length as its reserved KV "
                         "pool holds.  The p95 context length is the first known unknown of Section 5; "
                         "give the set of values this throughput takes over its documented range "
                         "(noise-free expected values).",
                   "tokens/s", key["lo"], key["hi"], tol.get("q2", 0.02 * key["hi"]),
                   floor=0.004 * key["hi"], cards=["S1", "S5"], obstacles=["O13", "O2"],
                   chain=["capacity", "roofline", "extremes"]),
        Q.cex("q3", "Is the following forced by the evidence?  Free parameters: %s" % ranges,
              "Raising the draft scheduler's proposal length from %d to %d, with the draft model, the "
              "hardware and the serving configuration unchanged, raises steady-state decode throughput."
              % (G_MAX, G_ASK),
              sch, entailed=False, z_bar=ZB_GRADE,
              cards=["S3"], obstacles=["O11", "O13"], chain=["profile", "tail"]),
        Q.cex("q4", "Is the following forced by the evidence?  Free parameters: %s" % ranges,
              "Raising the draft scheduler's proposal length from %d to %d, with everything else "
              "unchanged, cannot raise steady-state decode throughput by more than %.2f%%."
              % (G_MAX, G_ASK, 100.0 * p["x4"]),
              sch, entailed=True, z_bar=ZB_GRADE,
              cards=["S3"], obstacles=["O11"], chain=["profile", "bound"]),
    ]


def answers_from(ph, ctx, collapse=False):
    out = json.loads(json.dumps(ph.get("answers", ph)))
    if collapse:
        for a in out.values():
            if isinstance(a, dict) and "lo" in a and a["hi"] > a["lo"]:
                a["lo"] = a["hi"] = 0.5 * (a["lo"] + a["hi"])
    return out


# ----------------------------------------------------------------------------------------- oracle
def oracle_design(cap, pf):
    """A matched `spec_g=0` / `spec_g=G_MAX` pair at the benchmarking configuration - the only place the
    decode step cancels out of the ratio - bought long, since the acceptance positions and the two rate
    channels are what set q1's precision; and the two production points measured directly, which the
    reserved pool being smaller than the free memory is what makes possible."""
    sq, bt = int(pf["bench_seq"]), int(pf["bench_batch"])
    reqs = [{"seq": sq, "batch": bt, "draft": "none", "spec_g": 0, "dur": DUR_NB, "seed": 300 + k}
            for k in range(R_Q1)]
    reqs += [{"seq": sq, "batch": bt, "draft": DRAFT, "spec_g": G_MAX, "dur": DUR_NB, "seed": 310 + k}
             for k in range(R_Q1)]
    reqs += [{"seq": cap["slo"], "batch": cap["bh"], "draft": "none", "spec_g": 0, "dur": DUR_CAP,
              "seed": 320 + k} for k in range(R_Q2)]
    reqs += [{"seq": cap["shi"], "batch": cap["bl"], "draft": "none", "spec_g": 0, "dur": DUR_CAP,
              "seed": 330 + k} for k in range(R_Q2)]
    return reqs


def _est_from_rows(rows, cap):
    """The two numeric answers, from the oracle's rows.

    The step-time ratio is read in both rate channels and combined by inverse variance in log space:
    `tokens_per_s` carries sig_tput and `ms_per_token` sig_lat, they are independent draws on the same
    rows, and using only one of them throws away a third of the precision for nothing."""
    spec_r = [r for r in rows if int(r["config"].get("spec_g", 0)) == G_MAX]
    sq, bt = int(spec_r[0]["config"]["seq"]), int(spec_r[0]["config"]["batch"])
    base_r = [r for r in rows if int(r["config"].get("spec_g", 0)) == 0
              and int(r["config"]["seq"]) == sq and int(r["config"]["batch"]) == bt]
    es = []
    for r in spec_r:                                   # E4 from the measured per-position profile
        e, pre = 1.0, 1.0
        for a in r["accept_by_position"]:
            pre *= a
            e += pre
        es.append(e)
    E4 = float(np.mean(es))
    lm = lambda rs, k: float(np.mean([math.log(r[k]) for r in rs]))
    r_lat = lm(spec_r, "ms_per_token") - lm(base_r, "ms_per_token")      # log((1 + G_MAX c) / E4)
    r_tpt = lm(base_r, "tokens_per_s") - lm(spec_r, "tokens_per_s")      # the same quantity
    wl, wt = 1.0 / SV.BASE["sig_lat"] ** 2, 1.0 / SV.BASE["sig_tput"] ** 2
    step = math.exp((wl * r_lat + wt * r_tpt) / (wl + wt)) * E4          # = 1 + G_MAX * c
    sel = lambda s, b: [r for r in rows if int(r["config"]["seq"]) == s and int(r["config"]["batch"]) == b
                        and int(r["config"].get("spec_g", 0)) == 0]
    gm = lambda rs: math.exp(float(np.mean([math.log(r["tokens_per_s"]) for r in rs])))
    return {"f": 1.0 - 1.0 / step,
            "lo": gm(sel(cap["shi"], cap["bl"])), "hi": gm(sel(cap["slo"], cap["bh"]))}


_WIT = {}


def _fit_witness(pf, rows, box):
    """The witness for q3: the acceptance constants that best reproduce the notebook, with the tail at the
    bottom of its documented range.

    Fitting to the *disclosed rows* rather than to the truth is what makes the witness robust.  It is the
    same fit the agent performs; it clears the chi-square half of the criterion by construction rather than
    by luck; and it stays valid when the agent's own runs are appended to the evidence, because those runs
    are draws from the same world and only tighten the band the fit already sits in the middle of.  The
    score adds a large penalty for failing consistency and a smaller one for failing to reverse the claim,
    so the search returns a point that provably does both whenever one exists."""
    ckey = (id(rows), round(box["a"][0], 6), round(box["rho"][0], 6))
    if ckey in _WIT:
        return dict(_WIT[ckey])
    from .. import verify as V
    base = SV.full(pf)
    c = float(base["drafts"][DRAFT]["c"])
    (alo, ahi), (rlo, rhi) = box["a"], box["rho"]
    best, arg = None, (float(base["drafts"][DRAFT]["a"]), float(base["drafts"][DRAFT]["rho"]))
    for span, n in ((3.0, 13), (1.0, 9), (0.35, 9)):
        a0, r0 = arg
        for i in range(n):
            a = a0 + span * box["ha"] * (2.0 * i / (n - 1.0) - 1.0)
            if not alo <= a <= ahi:
                continue
            for j in range(n):
                r = r0 + span * box["hr"] * (2.0 * j / (n - 1.0) - 1.0)
                if not rlo <= r <= rhi:
                    continue
                q = json.loads(json.dumps(base))
                q["drafts"][DRAFT].update(a=a, rho=r, rho_tail=RT_LO)
                ok, det = V.consistency(SV, q, rows, z_bar=ZB_GRADE)
                s = (det.get("sum_z2", 1e18) + (0.0 if ok else 1e12)
                     + (0.0 if _ratio(a, r, c, RT_LO) <= 1.0 else 1e9))
                if best is None or s < best:
                    best, arg = s, (a, r)
    if best is not None and best >= 1e9:
        # No point on the grid both fits and flips.  The world itself always does: `wellposed` certifies
        # that the true parameters reproduce the notebook, and `_admissible` certifies that the claim is
        # false at the bottom of the documented tail range.  A witness world differs from the true one only
        # in a constant no observable depends on, so it reproduces *any* evidence the agent can gather.
        arg = (float(base["drafts"][DRAFT]["a"]), float(base["drafts"][DRAFT]["rho"]))
    out = {"a": round(arg[0], 6), "rho": round(arg[1], 6), "rho_tail": RT_LO}
    _WIT[ckey] = out
    return dict(out)


def _corner(box):
    """The assignment an agent tries first: every constant at the top of its range.  G14 certifies that it
    is inconsistent with the evidence, so a rival that answers `refutable` with it fails on the witness."""
    return {"a": box["a"][1], "rho": box["rho"][1], "rho_tail": RT_HI}


def _verdicts(pf):
    """How a world answers the two counterexample questions, reading the tail only through its documented
    range.  In the true world this reproduces the certified verdicts; in a world with a card neutralised it
    is what an analyst holding *that* model concludes."""
    d = pf["drafts"][DRAFT]
    a, rho, c = float(d["a"]), float(d["rho"]), float(d["c"])
    return ("entailed" if _ratio(a, rho, c, RT_LO) > 1.0 else "refutable",
            "entailed" if _ratio(a, rho, c, RT_HI) <= 1.0 + float(pf["x4"]) else "refutable")


def _ans(f, lo, hi, v3, v4, wit, corner):
    a3 = {"verdict": v3}
    if v3 == "refutable":
        a3["witness"] = wit
    a4 = {"verdict": v4}
    if v4 == "refutable":
        a4["witness"] = corner
    return {"q1": {"lo": f, "hi": f}, "q2": {"lo": lo, "hi": hi}, "q3": a3, "q4": a4}


def oracle(sess, rows_nb, ctx, rng, drop=None):
    pf = SV.full(sess.p)
    box = _box(pf["drafts"][DRAFT]["a"], pf["drafts"][DRAFT]["rho"])
    wit, corner = _fit_witness(pf, rows_nb, box), _corner(box)
    if drop:
        # G9 asks whether a mechanism is load-bearing for the *answers*.  Neutralise it in the world and
        # recompute the keys exactly, so that a kill is the mechanism and never a bad draw.  A neutralised
        # world generally has no consistent fit to the disclosed profile at all, so the two counterexample
        # questions are answered the way an analyst holding the neutralised model answers them: by
        # evaluating the claim over the documented tail range under that model.
        p2 = SV.full(json.loads(json.dumps(pf)))
        for path, v in drop.items():
            _set(p2, path, v)
        k = _keys(p2)
        v3, v4 = _verdicts(p2)
        ans = _ans(k["f"], k["lo"], k["hi"], v3, v4, wit, corner)
        return ans, {"answers": ans, "drop": sorted(drop)}
    cap = _cap(pf)
    rows = run_rows(sess, oracle_design(cap, pf))
    est = _est_from_rows(rows, cap)
    ans = _ans(est["f"], est["lo"], est["hi"], "refutable", "entailed", wit, corner)
    return ans, {"answers": ans, "n_rows": len(rows), "witness": wit}


def cert_requests(spec_, rng):
    """`spec_g` at the build's cap is the only place the tail decay could possibly show up, and the random
    certificate sample spreads its draws over five knobs, so those requests are made explicit here: every
    proposal length the build allows, across the context range.  `batch=1` always fits."""
    return [{"seq": s, "batch": 1, "draft": DRAFT, "spec_g": g, "dur": DUR_NB, "seed": 7}
            for g in range(G_MAX + 1) for s in (256, 1024, 4096)]


# ----------------------------------------------------------------------------------------- rivals
def rivals(p, rows_nb, ctx, rng):
    pf = SV.full(p)
    cap = _cap(pf)
    key = _keys(pf)
    box = _box(pf["drafts"][DRAFT]["a"], pf["drafts"][DRAFT]["rho"])
    wit, corner = _fit_witness(pf, rows_nb, box), _corner(box)
    a, rho, c = (float(pf["drafts"][DRAFT][k]) for k in ("a", "rho", "c"))
    E4 = _E(a, rho, G_MAX)
    step = 1.0 + G_MAX * c                      # the quantity q1 is a function of
    out = {}

    def reg(name, f=key["f"], lo=key["lo"], hi=key["hi"], v3="refutable", v4="entailed"):
        out[name] = _ans(f, lo, hi, v3, v4, wit, corner)

    # the published draft-cost ratio in place of a measurement of this lab's draft model
    reg("B_prior", f=G_MAX * C_PUB / (1.0 + G_MAX * C_PUB))

    # the memo's reading: the capacity probe's `ms_per_token` used as the non-speculative baseline, though
    # it was measured at a different context and batch, where the decode step is a different length
    m = (SV.step_time(pf, BITS, int(pf["bench_seq"]), int(pf["bench_batch"]))
         / SV.step_time(pf, BITS, cap["shi"], cap["bl"]))
    reg("skip:matched", f=1.0 - 1.0 / (step * m))

    # `accept_rate` (the mean of the measured positions) taken as *the* acceptance rate, so the expected
    # accepted tokens per step are those of a flat profile - and the tail is then flat as well
    reg("skip:profile", f=1.0 - 1.0 / (step * _E_flat(_flat_a(a, rho), G_MAX) / E4),
        v3="entailed", v4="refutable")

    # `ms_per_token` read as the verification-step time, so the step ratio is taken to be ms4/ms0 directly.
    # The arithmetic then returns a negative drafting cost; the shortcut is not noticing that.
    reg("skip:perstep", f=1.0 - E4 / step)

    # the lab box's own `batch_max` used as the production concurrency, instead of the reserved KV pool
    reg("skip:capacity", lo=cap["box_lo"])

    # the memory-traffic formula applied at the short context too, missing the compute roof
    reg("skip:roofline", hi=cap["mem_hi"])

    # the context range collapsed to its midpoint, as the memo's capacity plan proposes
    reg("skip:extremes", lo=cap["kmid"], hi=cap["kmid"])

    # the decay assumed to continue past the measured positions at the same rate: the known unknown denied
    reg("skip:tail", v3="entailed")

    # "the tail cannot be measured, so nothing can be bounded": the entailed bound answered `refutable`
    reg("skip:bound", v4="refutable")

    # G12 names a nuisance card and G11 a reasoning step; where the two are the same shortcut seen from
    # different sides the same answers are registered under both names, rather than a rival being invented.
    for card, step_name in (("S3", "profile"), ("S5", "capacity"), ("S1", "roofline")):
        out["naive_ignore:" + card] = out["skip:" + step_name]
    return out


def rival_designs(p, rows_nb, rng):
    return {}


DROP = {"S1": {"P_peak": 1.0e18},
        # neutralising speculation means switching off its cost as well as its yield: with c = 0 the
        # fraction of the step that goes to drafting is exactly zero, which is where q1's kill comes from
        "S3": {"drafts.%s.c" % DRAFT: 0.0, "drafts.%s.rho" % DRAFT: 1.0,
               "drafts.%s.rho_tail" % DRAFT: 1.0},
        # not 0: `batch_max` divides by the per-token KV bytes
        "S5": {"kv_bytes": 0.05}}
INFO_RIVALS = ()


# -------------------------------------------------------------------------------------- posedness
def _admissible(pf):
    """Thresholds are this task's tolerance arithmetic, not taste.

    `calibrate` sets T = 2.25 * p90(|oracle error|).  q2's endpoints are measured directly at `dur=20`
    with R_Q2 repetitions, a relative spread of sig_tput / sqrt(dur/20 * reps) ~ 0.6%, which puts T near
    1.7% of the upper endpoint; G3 wants the set at least 4T ~ 7% wide, hence W_MIN = 1.22 with better
    than two-fold headroom.  q1 combines two rate channels over R_Q1 repetitions and the profile, about
    0.4 points of a per cent, so T ~ 0.012 and a rival has to move the fraction by 0.024: GAP1_MIN =
    0.040 is that with headroom, and PRIOR_MIN = 0.10 is the eight sigma G13 asks for on the prior."""
    d = pf["drafts"][DRAFT]
    a, rho, c, rt = float(d["a"]), float(d["rho"]), float(d["c"]), float(d["rho_tail"])
    cap = _cap(pf)
    w = cap["khi"] / max(cap["klo"], 1e-9)
    roof = cap["mem_hi"] / max(cap["khi"], 1e-9)
    ex = min(cap["khi"] / max(cap["kmid"], 1e-9) - 1.0, 1.0 - cap["klo"] / max(cap["kmid"], 1e-9))
    capr = cap["box_lo"] / max(cap["klo"], 1e-9)
    t_b = SV.step_time(pf, BITS, int(pf["bench_seq"]), int(pf["bench_batch"]))
    t_p = SV.step_time(pf, BITS, cap["shi"], cap["bl"])
    E4 = _E(a, rho, G_MAX)
    step = 1.0 + G_MAX * c
    f = 1.0 - 1.0 / step
    f_flat = 1.0 - 1.0 / (step * _E_flat(_flat_a(a, rho), G_MAX) / E4)
    fp = G_MAX * C_PUB / (1.0 + G_MAX * C_PUB)
    rts = _rts(a, rho, c)
    x4, sup, flat, loose = _bound(a, rho, c)
    box = _box(a, rho)
    vol = _vol_proxy(a, rho, c, box)
    cont = _ratio(a, rho, c, rho)
    rev = _ratio(a, rho, c, RT_LO)
    checks = [
        (cap["bl"] >= 12, "the production replica holds only %d sequences at the long context" % cap["bl"]),
        (cap["bh"] <= cap["bmax_lo"],
         "the short-context production point (batch %d) does not fit on the lab box (batch_max %d)"
         % (cap["bh"], cap["bmax_lo"])),
        (abs(cap["khi"] / cap["plateau"] - 1.0) < 1e-9,
         "the short end must sit on the compute roof: %.4f of the plateau" % (cap["khi"] / cap["plateau"])),
        (cap["klo"] <= 0.999 * cap["plateau"],
         "the long end must be memory-bound: %.4f of the plateau" % (cap["klo"] / cap["plateau"])),
        (W_MIN <= w <= W_MAX, "q2 set width khi/klo = %.4f outside [%.2f, %.2f]" % (w, W_MIN, W_MAX)),
        (roof >= ROOF_MIN, "the compute roof must bite: the memory formula is only %.4fx the truth" % roof),
        (ex >= EX_MIN, "a midpoint answer must miss both ends: shortfall %.4f" % ex),
        (capr >= CAP_MIN, "the lab box's batch_max must overstate production capacity: %.4fx" % capr),
        (abs(t_b / t_p - 1.0) >= MATCH_MIN,
         "the memo's baseline sits at the same step length: ratio %.4f" % (t_b / t_p)),
        (int(pf["bench_batch"]) <= SV.batch_max(pf, BITS, int(pf["bench_seq"])),
         "the benchmarking configuration does not fit in memory"),
        (f - f_flat >= GAP1_MIN, "a flat acceptance profile must move q1: %.4f against %.4f" % (f, f_flat)),
        (abs(f - fp) >= PRIOR_MIN,
         "q1's anti-prior gap is %.4f (this lab %.4f, the published constant %.4f)" % (abs(f - fp), f, fp)),
        (RTS_LO <= rts <= min(RTS_HI, rho - RTS_PAD),
         "the tie point rho_tail = %.4f is outside [%.3f, %.3f]"
         % (rts, RTS_LO, min(RTS_HI, rho - RTS_PAD))),
        (rts < rt < RT_HI, "the true tail %.4f must lie above the tie point %.4f" % (rt, rts)),
        (cont >= CONT_MIN, "assuming the decay continues must force the claim: ratio %.4f" % cont),
        (rev <= REV_MAX,
         "the bottom of the documented tail range must reverse the claim: ratio %.4f" % rev),
        (min(flat, loose) / sup >= ROOM_MIN,
         "no room for the entailed bound: sup %.4f, flat %.4f, loose %.4f" % (sup, flat, loose)),
        (VOL_LO <= vol <= VOL_HI, "witness volume %.4f outside [%.3f, %.3f]" % (vol, VOL_LO, VOL_HI)),
        (box["a"][1] < 1.0 and box["rho"][1] < 1.0, "the declared box runs past acceptance 1"),
    ]
    for ok, why in checks:
        if not ok:
            return False, why
    return True, ("q2 %.0f-%.0f tok/s (width %.3fx; the memory formula would say %.0f, the lab box's "
                  "batch_max %.0f, the midpoint %.0f); q1 %.4f (flat profile %.4f, published constant "
                  "%.4f, memo's baseline %.4f); tie point %.4f, true tail %.4f, continuation %.4f, bottom "
                  "%.4f; bound %.4f between sup %.4f and min(flat %.4f, loose %.4f); witness volume %.4f"
                  % (cap["klo"], cap["khi"], w, cap["mem_hi"], cap["box_lo"], cap["kmid"],
                     f, f_flat, fp, 1.0 - 1.0 / (step * t_b / t_p),
                     rts, rt, cont, rev, x4, sup, flat, loose, vol))


def _vol_real(pf, rows, box, n=700, seed=11):
    """G14's own question, asked with G14's own criterion: of the assignments in the declared box, what
    share both reproduce *these* rows - chi-square included - and reverse the claim of q3."""
    from .. import verify as V
    rng = np.random.default_rng(seed)
    base = SV.full(pf)
    c = float(base["drafts"][DRAFT]["c"])
    (alo, ahi), (rlo, rhi) = box["a"], box["rho"]
    nb = 0
    for _ in range(n):
        a, r = float(rng.uniform(alo, ahi)), float(rng.uniform(rlo, rhi))
        rt = float(rng.uniform(RT_LO, RT_HI))
        if _ratio(a, r, c, rt) > 1.0:
            continue
        q = json.loads(json.dumps(base))
        q["drafts"][DRAFT].update(a=a, rho=r, rho_tail=rt)
        if V.consistency(SV, q, rows, z_bar=ZB_GRADE)[0]:
            nb += 1
    return nb / float(n)


MAY_COLLIDE = ("param:drafts.%s.a" % DRAFT,)


def _stray_collisions(w):
    """Which hidden constants the acceptance exemption is covering for, beyond the one it argues for.

    `public_values` is a list of *numbers*, not of names, so declaring a measured acceptance of 0.8996
    exempts anything else in the world that renders as 0.8996 - and `mem_util` lives in a range that
    collides with the acceptance range often enough to matter (three times in a hundred and twenty draws).
    That is a real hole: the argument for the exemption is that a disclosed estimate of a constant does not
    leak the constant, and it does not extend to an unrelated parameter that happens to print the same.
    So the scan is re-run here with the exemption withdrawn, and any hit that is not the acceptance
    constant itself rejects the draw.  Cheaper than making the package's exemptions name-keyed, and it
    fails closed: a new collision class shows up as a lost seed, not as a silent disclosure."""
    from .. import build as _B
    keep = set(map(id, w["ctx"]["acc_obs"]))
    pub = [v for v in public_values(w["ctx"]) if id(v) not in keep]
    text = w["notes"] + json.dumps(w["rows"]) + "\n".join(w["files"].values())
    hits = _B.leakage_scan(w, [], text, pub)["hits"]
    return [h for h in hits if h["what"] not in MAY_COLLIDE]


def wellposed(w):
    """Admissibility of the world, and of the evidence drawn in it.

    The second half is here for the counterexample questions.  With twenty-odd observables the notebook has
    about a one-in-twenty chance of drawing a row its own world sits three sigma from, and in that instance
    *no* witness reproduces the evidence, because the witness has to fit the same rows; and the size of the
    witness set is a property of the drawn rows, not only of the world.  Both are cheap to check here and
    become opaque oracle failures if left to G1."""
    from .. import verify as V
    ok, why = _admissible(w["pf"])
    if not ok:
        return False, why
    stray = _stray_collisions(w)
    if stray:
        return False, ("a hidden constant is legible in the shipped prose: "
                       + ", ".join("%s as %s" % (h["what"], h["pattern"]) for h in stray[:3]))
    pf = SV.full(w["pf"])
    c, det = V.consistency(SV, pf, w["rows"], z_bar=ZB_GRADE)
    if not c or det["max_abs_z"] > 0.80 * ZB_GRADE or det["sum_z2"] > 0.85 * det["chi2_bar"]:
        return False, ("the drawn notebook is too far from its own world for a witness to fit it: "
                       "max |z| %.2f, sum z^2 %.1f of %.1f"
                       % (det["max_abs_z"], det["sum_z2"], det["chi2_bar"]))
    box = _box(pf["drafts"][DRAFT]["a"], pf["drafts"][DRAFT]["rho"])
    rv = _vol_real(pf, w["rows"], box)
    if not RVOL_LO <= rv <= RVOL_HI:
        return False, ("witness volume against the drawn rows is %.4f, outside [%.3f, %.3f]"
                       % (rv, RVOL_LO, RVOL_HI))
    return True, why + "; notebook max |z| %.2f, sum z^2 %.1f of %.1f, witness volume %.4f" % (
        det["max_abs_z"], det["sum_z2"], det["chi2_bar"], rv)


def public_values(ctx):
    vals = list(ctx["pub"]) + list(ctx["obs"]) + list(ctx["prof"])
    cap = ctx["cap"]
    vals += [cap["slo"], cap["shi"], cap["bl"], cap["bh"], cap["bmax_lo"], cap["bmax_hi"], cap["kmid"],
             ctx["ms_spec"], ctx["tp_spec"], ctx["acc"], ctx["speed"], ctx["ms_probe"], ctx["tp_probe"],
             ctx["e8f"], ctx["e4f"], ctx["c_share"]]
    # Every acceptance reading the notebook prints is disclosed by construction - it is the measurement the
    # memo argues from - and at three or four significant figures one of the twelve of them collides with
    # the hidden acceptance constant in a noticeable share of instances (five draws in a hundred and
    # twenty).  A disclosed *estimate* of a constant is not a leak of the constant (the agent can make the
    # same measurement for the price of one run), so the column is declared rather than left to chance.
    # The exemption is a list of numbers and so is blind to *which* parameter it covers; `_stray_collisions`
    # narrows it back to the acceptance constant and rejects any draw where it would cover anything else.
    vals += list(ctx["acc_obs"])
    return vals
