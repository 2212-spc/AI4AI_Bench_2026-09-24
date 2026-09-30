"""Mixed-tenant tail latency: the p99 a six-tenant traffic mix actually serves, next quarter's growth, and
which production batch to ship.

This is the first blueprint written *from* the v5 difficulty law rather than before it, so the design rule
is worth stating.  `exp/sweep_B.py` measured

    B  =  bias_geom / (2.25 * kappa * sigma / sqrt(n))

where B is the discrimination of an item (how many tolerances separate the best shortcut from the truth),
`bias_geom` is the shortcut's structural error, and `kappa` is the conditioning of the *reference*
estimator.  The two geometry knobs an author naturally reaches for - push the question further from what the
lab sells, curve the response harder - raise `bias_geom` five- to sevenfold and `kappa` faster, so they are
falsified: `exp/kappa_price.json` prices B = 3 on servelab's saturation-rate estimand at 173 runs against
2 runs for its aggregation estimand, a factor of 73 = kappa^2.  Worse, the saturation item's shortcut bias
is 2.6% of the truth against a 1% reporting floor, so its B *ceiling* is 2.6: no budget whatsoever makes it
discriminating.

The rule that survives is therefore: **a well-conditioned oracle (kappa ~ 1: interpolate, average, compare
matched groups - never extrapolate-and-invert) plus an O(1) shortcut bias controlled by an independent world
constant.**  Every question here is built that way.  The independent constant is S4's `cs2`; the shortcut
bias is Jensen's gap, which is large because E[T] diverges as utilisation approaches one, so the tail a
mixed fleet serves is set by its *busy* tenants and not by its mean rate.  The oracle is an inverse-variance
average of a quantity that enters the response linearly, so kappa = 1 exactly and the measured B is 15.8 to
27.6 across twelve draws (`exp/probe_s01.py`), where v4's best servelab item could not reach 3 at any price.

Two honest disclosures about what this blueprint does *not* do.

First, its difficulty axes are **D (must the estimator be invented?) and B (is the shortcut separated?),
not rho (is the budget binding?)**.  Section 2 of the manual guarantees `util` is exact, so a single `load`
row hands over E[S] = util/rate with no noise at all, and the whole design of the questions is visible from
the notebook.  What the agent cannot get for free is precision on `cs2`: the notebook's three short rows pin
the answer only to about 9% - measured at 7.4 to 8.4 tolerances - so the item is not free, and matching the
reference estimator's precision costs about 0.72 of the total budget.  That is a real rho, but it is bought
with the *oracle's* precision rather than with the lab's refusals, which is a weaker mechanism than a hard
ceiling and is recorded as such.

Second, **S1 is deliberately not claimed.**  `service_time` does carry a prefill term in `P_peak` and
`prefill_flop_eff`, so neutralising those constants would move the answers - but the *mechanism* S1 names is
the roofline switch (a decode step costing the larger of its memory and arithmetic time), and that switch is
provably inert at production context lengths here: the draw keeps `kv_per_token * seq / BW` above
`2N/(P_peak * eff)`, so memory binds at every batch the questions mention and `b_crit`'s reported figure is
a red herring (it ignores KV traffic).  Claiming S1 would pass G9 for the wrong reason - the constants, not
the mechanism - so CARDS names only what is load-bearing as a mechanism.

Useful designs: a `load` row's `util` is exact, so one row at any stable rate gives E[S] at that batch for
free; `p50_ms` and `p99_ms` are *independent* draws on the same row, and since p50/p99 = ln2/ln100 exactly,
combining both channels buys a factor sqrt(2) of precision for nothing.  The variability constant enters
E[T]/E[S] - 1 linearly in rho/(1-rho), so it is best measured at the *busiest* buckets the fleet actually
runs, and `dur` there is the only precision knob that matters.
"""
import json
import math

import numpy as np

from .. import queries as Q
from ..common import draw_card, run_rows
from ..labs import servelab as SV

ID = "s01-mix-p99"
TITLE = "Mixed-tenant tail latency: the p99 a fleet serves, what growth does to it, and which batch to ship"
CARDS = ["S4", "S5"]
OBSTACLES = ["O14", "O2", "O13", "O6"]
CLAIM = ("recover the tail latency a heterogeneous traffic mix actually serves, rather than the tail at its "
         "mean arrival rate, and carry that aggregate through a disclosed growth range and a feasibility "
         "screen on the batch the fleet can deploy")
DIFFICULTY = {"depth": 3, "nuisance": ["S4", "S5"], "anti_prior": ["q1"]}
EXEMPT_LOAD_BEARING = {}
CERT_N = 400

BITS = 16

# The six tenants' utilisations at the production batch, as a ladder.  Utilisation is *not* a knob (S4), so
# these are realised by solving for arrival rates at the production E[S]; the ladder is what makes Jensen's
# gap large - 1 + k*rho/(1-rho) is convex and the top of the ladder dominates the mean.
RHO_T = (0.25, 0.40, 0.55, 0.70, 0.80, 0.85)
JIT = 0.010                     # per-tenant jitter, so the ladder is not a round-number giveaway
GROWTH = 1.03                   # disclosed: every tenant's rate grows by this much next quarter
TOP_HI = 1.08                   # the top tenant's *committed peak* - the known unknown's upper end
RHO_NEXT_MAX = 0.93             # next quarter must not saturate the top tenant, or q2's upper end is infinite
W_MIN = 1.20                    # minimum Jensen ratio truth/naive (see `_admissible`)

LN100 = math.log(100.0)
LN2 = math.log(2.0)
FAIL_MS = 10000.0               # the loss charged to an inadmissible option, stated in q3's text

# Oracle design.  `cs2` enters E[T]/E[S] - 1 linearly in x = rho/(1-rho), so the per-row precision on it
# scales as x, and the three busiest buckets are worth far more than six spread evenly: concentrating the
# same cost there took T from 2.4% of the truth to 1.19%, which is what puts rho near 0.7 rather than 0.2.
OR_IDX, OR_REPS, OR_DUR = (3, 4, 5), 4, 30.0
MENU_DUR = 5.0                  # the feasibility rows only need `util`, which is exact at any duration
NB_REPS, NB_DUR = 3, 5.0        # the notebook's own load probe: short, so it is not a free answer


def _set(p, path, v):
    parts = path.split(".")
    node = p
    for s in parts[:-1]:
        node = node[s]
    node[parts[-1]] = v


# --------------------------------------------------------------------------------- the estimand, in closed form
def _es(pf, seq, batch):
    return SV.service_time(pf, BITS, seq, batch)


def _factor(k, rho):
    """E[T]/E[S] for M/G/1 with k = (1 + cs2)/2.  None when the queue does not clear."""
    if rho >= 1.0:
        return None
    return 1.0 + k * rho / (1.0 - rho)


def _mix_p99(pf, seq, batch, rates, k=None):
    """The equally-weighted mean, over the tenants, of the p99 *each tenant* experiences.

    Not the p99 of the pooled latency distribution, which is a different and larger quantity: every tenant
    is an independent M/G/1 stream here, and the question is worded to ask for the mean of their tails."""
    k = 0.5 * (1.0 + pf["cs2"]) if k is None else k
    S = _es(pf, seq, batch)
    acc = 0.0
    for r in rates:
        f = _factor(k, r * S)
        if f is None:
            return None
        acc += f
    return 1e3 * LN100 * S * acc / len(rates)


def _naive_mean_rate(pf, seq, batch, rates, k=None):
    """The Jensen shortcut: the p99 at the mix's *mean* arrival rate.  This is the memo's reading and, at a
    ladder that reaches rho = 0.85, understates the served tail by 20 to 50 per cent."""
    k = 0.5 * (1.0 + pf["cs2"]) if k is None else k
    S = _es(pf, seq, batch)
    f = _factor(k, float(np.mean(rates)) * S)
    return None if f is None else 1e3 * LN100 * S * f


def _b_fit(pf, seq, pool):
    """How many whole sequences of the production context the *reserved* KV pool admits (S5).

    This is the constraint the memo never checks: the development box's `batch_max` is much larger, so a
    batch can benchmark beautifully here and be undeployable there."""
    return int(pool // (SV.kv_per_token(pf) * float(seq)))


def _rates_next(p, top):
    """Next quarter's arrival rates: everyone grows by the disclosed GROWTH, the top tenant to `top`."""
    r = p["tenant_rates"]
    return [GROWTH * x for x in r[:-1]] + [top * r[-1]]


# --------------------------------------------------------------------------------------------------- draw
def draw(rng):
    """Constructive, not rejection-first: four windows have to hold simultaneously and blind draws miss.

    The utilisation ladder is chosen first and the arrival rates are *solved* from the production E[S],
    because utilisation is not a knob.  Then the menu is checked to be genuinely discriminating (one option
    unstable, one feasible but over the SLO, one best, one that only the development box can run), the
    reserved pool is placed to cut off the largest menu batch, and the SLO is placed between the naive
    reading and the truth so that the Jensen gap is a decision and not just a number."""
    why = "no draw attempted"
    for _ in range(400):
        N = float(np.exp(rng.uniform(math.log(6.5e9), math.log(1.4e10))))
        p = dict(N=N, L=int(rng.integers(28, 49)), H_kv=8, d_head=128, M_gpu=80.0e9,
                 act_bytes=float(rng.uniform(1.4e9, 2.6e9)),
                 P_peak=float(rng.uniform(3.4e14, 4.6e14)), BW=float(rng.uniform(2.8e12, 4.2e12)))
        p.update(draw_card(rng, "S1", {"eff": (0.55, 0.78)}))
        p.update(draw_card(rng, "S5", {"mem_util": (0.84, 0.92), "kv_bytes": (1.9, 2.0)}))
        p.update(draw_card(rng, "S4", {"cs2": (1.6, 2.9)}))
        pf = SV.full(p)

        B0 = int(rng.choice([24, 32, 40]))
        menu = (B0 // 2, B0, B0 + 16, 3 * B0)
        seq = 128 * int(rng.integers(12, 21))                    # 1536 .. 2560
        if SV.batch_max(pf, BITS, seq) < menu[3] + 8:
            why = "the development box cannot run the largest menu batch at seq=%d" % seq
            continue
        S0 = _es(pf, seq, B0)
        rates = [round((t + float(rng.uniform(-JIT, JIT))) / S0, 2) for t in RHO_T]
        if len(set(rates)) < len(rates) or min(rates) <= 0:
            why = "tenant rates collide after rounding to two decimals"
            continue
        rho = [r * S0 for r in rates]
        if not all(x < y for x, y in zip(rho, rho[1:])):
            why = "rounding broke the tenant ordering"
            continue
        if max(rho) >= 0.88 or min(rho) <= 0.20:
            why = "tenant utilisation window broken (max %.3f, min %.3f)" % (max(rho), min(rho))
            continue
        # the menu must be a real decision: the small batch cannot serve the top tenant at all, and the
        # winning batch must relieve it enough to be visibly better rather than marginally
        if _es(pf, seq, menu[0]) * max(rates) <= 1.02:
            why = "batch %d is still stable for the top tenant" % menu[0]
            continue
        if _es(pf, seq, menu[2]) * max(rates) >= 0.88:
            why = "batch %d does not relieve the top tenant enough" % menu[2]
            continue
        if TOP_HI * rates[-1] * S0 >= RHO_NEXT_MAX:
            why = "next quarter saturates the top tenant (rho %.3f)" % (TOP_HI * rates[-1] * S0)
            continue

        # the reserved pool: it admits the winning batch and cuts off the largest one, and it is placed off
        # a whole-sequence boundary so the floor division is not reconstructible from a round number
        per = SV.kv_per_token(pf) * float(seq)
        lo_fit, hi_fit = menu[2] + 2, menu[3] - 6
        if lo_fit >= hi_fit:
            why = "no window for the reserved pool between the menu batches"
            continue
        fit = int(rng.integers(lo_fit, hi_fit + 1))
        pool = per * (fit + float(rng.uniform(0.15, 0.85)))
        if _b_fit(pf, seq, pool) != fit:
            why = "pool rounding"
            continue

        p.update(prod_seq=seq, prod_batch=B0, nb_batch=2 * B0, kv_pool=pool,
                 tenant_rates=rates, menu=list(menu),
                 # the scenario unknown: the top tenant's committed peak next quarter.  Nothing in the lab
                 # reads it - `full()` copies BASE and then updates, so a blueprint-private key survives
                 # into the session without any service consulting it.
                 top_rate_next=float(0.5 * (GROWTH + TOP_HI) * rates[-1]))
        pf = SV.full(p)

        truth = _mix_p99(pf, seq, B0, rates)
        naive = _naive_mean_rate(pf, seq, B0, rates)
        if truth is None or naive is None or truth / naive < W_MIN:
            why = "Jensen gap too small (%.3f)" % (truth / naive if naive else float("nan"))
            continue
        best = _mix_p99(pf, seq, menu[2], rates)
        # the SLO sits between the two readings, at the geometric mean rounded to a round number an SRE
        # would actually write down, so the analyst who takes the mean rate concludes "compliant" and the
        # analyst who aggregates correctly concludes "not compliant, and here is the batch that fixes it"
        slo = 50.0 * round(math.sqrt(truth * naive) / 50.0)
        if not (naive < 0.96 * slo and truth > 1.06 * slo and best < 0.90 * slo):
            why = "no SLO separating the naive reading from the truth"
            continue
        p["slo_ms"] = slo

        ok, why = _admissible(SV.full(p))
        if ok:
            return p
    raise RuntimeError("no admissible draw (last: %s)" % why)


# --------------------------------------------------------------------------------------------------- spec
def spec(p):
    return {"lab": "servelab",
            "knobs": {"svc": {"type": "choice", "values": ["bench", "load"], "default": "load"},
                      "seq": {"type": "float", "min": 256, "max": 4096, "int": True, "default": 1024},
                      "batch": {"type": "float", "min": 1, "max": 1024, "int": True, "default": 32},
                      "rate": {"type": "float", "min": 0.01, "max": 400.0, "default": 1.0},
                      "dur": {"type": "float", "min": 5.0, "max": 40.0, "default": 20.0}},
            "fixed": {"bits": BITS},
            # `dur` is the precision knob and the binding constraint.  The variability constant is what the
            # aggregate is sensitive to and it is measured best at the busiest buckets, so the reference
            # solution spends nearly all of its budget on long rows there: 17 runs and 385 of the 530.
            "caps": {"run_cost": 40.0, "total_cost": 530.0, "max_runs": 24}}


def LAB_EXTRA(w):
    pf = w["pf"]
    return ["",
            "This lab serves one model at `bits=16`.  `svc=load` is a closed-loop load test of **one tenant "
            "stream** at the arrival rate you give: it reports `util` (exact), `p50_ms` and `p99_ms`.  Each "
            "tenant in the production mix is an independent stream with its own arrival rate; the fleet does "
            "not pool them.",
            "",
            "`svc=bench` reports `kv_bytes_per_seq` and `batch_max`, both exact, and both describe **this "
            "development box**: `batch_max` is the largest batch whose weights, activations and KV cache fit "
            "in its memory.  The production replicas reserve a fixed KV pool of their own, declared in "
            "`/app/deploy/serving.yaml`, which is a different and smaller number (%.2f GB); a production "
            "batch is deployable only if that pool admits every sequence in it whole." % (pf["kv_pool"] / 1e9),
            "",
            "A `load` request whose arrival rate exceeds the service rate at that batch returns "
            "`status=unstable` with no latency figures, and is charged."]


def known_unknowns(p):
    r6 = p["tenant_rates"][-1]
    return [
        {"name": "the top tenant's committed peak rate next quarter",
         "param": "top_rate_next", "range": [GROWTH * r6, TOP_HI * r6],
         "text": "Every tenant's arrival rate grows by a disclosed %.0f%% next quarter.  The largest tenant "
                 "has in addition bought a *committed peak*, and its contract fixes only a band: its next "
                 "quarter rate lies between %.4f and %.4f requests per second (the low end is the %.0f%% "
                 "growth all tenants get, the high end is the peak it may draw).  Commercial has not settled "
                 "on a figure and nothing in this service depends on the choice.  A question about next "
                 "quarter must cover the whole band."
                 % (100.0 * (GROWTH - 1.0), GROWTH * r6, TOP_HI * r6, 100.0 * (GROWTH - 1.0))}]


# ----------------------------------------------------------------------------------------------- notebook
def _nb_reqs(p):
    """The team's own runs.  Three short `load` repetitions at the *mean* arrival rate and at the
    *development* batch - two compounding errors, each of which the memo then reasons from - plus two
    `bench` rows that make the larger batch look free."""
    seq, B0, nb = int(p["prod_seq"]), int(p["prod_batch"]), int(p["nb_batch"])
    mean_rate = round(float(np.mean(p["tenant_rates"])), 2)
    reqs = [{"svc": "load", "seq": seq, "batch": nb, "rate": mean_rate, "dur": NB_DUR, "seed": 100 + k}
            for k in range(NB_REPS)]
    reqs.append({"svc": "bench", "seq": seq, "batch": B0, "dur": NB_DUR, "seed": 200})
    reqs.append({"svc": "bench", "seq": seq, "batch": int(p["menu"][3]), "dur": NB_DUR, "seed": 201})
    return reqs


def notebook(p, sess, rng):
    pf = SV.full(p)
    rows = []
    for r in _nb_reqs(pf):
        cfg, seed, ex = sess.validate(dict(r))
        rows.append(sess.execute(cfg, seed, ex))
    load_rows, b_prod, b_big = rows[:NB_REPS], rows[NB_REPS], rows[NB_REPS + 1]
    p99 = float(np.mean([r["p99_ms"] for r in load_rows]))
    p50 = float(np.mean([r["p50_ms"] for r in load_rows]))
    util = float(load_rows[0]["util"])
    mean_rate = round(float(np.mean(pf["tenant_rates"])), 2)
    tp_ratio = b_big["tokens_per_s"] / b_prod["tokens_per_s"]
    notes = """# Serving review: tail latency and the production batch (inference platform)

We are being asked to sign off two things before the quarter closes: whether the fleet meets its %.0f ms p99
SLO, and which batch size the production replicas should run.  Our runs are in `notebook/runs.jsonl`; the
deployment we are sizing is `deploy/serving.yaml`.

- **Tenants.**  Six tenants share the fleet, each an independent stream.  Their measured arrival rates are
  %s requests per second, so the fleet averages **%.2f req/s**.  We load-tested at that average.
- **The tail looks fine.**  At `rate=%.2f`, `batch=%d`, `seq=%d` we measure `util` %.4f, `p50_ms` %.1f and
  `p99_ms` %.1f (three repetitions, `dur=%.0f`).  Against a %.0f ms SLO that is **comfortably under**, so
  the fleet is compliant on today's traffic and has room for the %.0f%% growth we have been told to plan for.
- **Batch.**  Throughput per replica keeps rising with batch: `tokens_per_s` is %.1f at `batch=%d` and %.1f
  at `batch=%d`, a **%.2fx** improvement, and both ran fine here (`batch_max` on this box is %d at this
  context).  **Recommendation: ship `batch=%d`.**
- **If latency ever becomes a problem** we can always drop the batch back down - smaller batches mean
  shorter service times per request, so the queue should drain faster.

Open questions nobody has answered: we load-tested one stream at the average rate rather than each tenant at
its own rate, and the top tenant's committed peak for next quarter is still a band rather than a number.
""" % (pf["slo_ms"],
       ", ".join("%.2f" % r for r in pf["tenant_rates"]), mean_rate,
       mean_rate, int(pf["nb_batch"]), int(pf["prod_seq"]), util, p50, p99, NB_DUR, pf["slo_ms"],
       100.0 * (GROWTH - 1.0),
       b_prod["tokens_per_s"], int(pf["prod_batch"]), b_big["tokens_per_s"], int(pf["menu"][3]),
       tp_ratio, int(b_prod["batch_max"]), int(pf["menu"][3]))
    ctx = {"rows": rows, "p99_nb": p99, "p50_nb": p50, "util_nb": util, "mean_rate": mean_rate,
           "tp_ratio": tp_ratio,
           "obs": [v for r in rows for v in (r.get("util"), r.get("rate"), r.get("p50_ms"), r.get("p99_ms"),
                                             r.get("tokens_per_s"), r.get("ms_per_token"),
                                             r.get("kv_bytes_per_seq"), r.get("batch_max"), r.get("batch"))
                   if v is not None],
           "pub": [pf["prod_seq"], pf["prod_batch"], pf["nb_batch"], pf["slo_ms"], pf["kv_pool"],
                   pf["kv_pool"] / 1e9, float(int(pf["kv_pool"])),
                   GROWTH, 100.0 * (GROWTH - 1.0), TOP_HI, NB_DUR, FAIL_MS,
                   GROWTH * pf["tenant_rates"][-1], TOP_HI * pf["tenant_rates"][-1]]
           + list(pf["tenant_rates"]) + [float(b) for b in pf["menu"]]}
    return rows, notes, ctx


def files(p, ctx, rng):
    yaml = """# production serving configuration (replica template) -- do not edit by hand
model:
  name: target-%dm
  dtype: float16
runtime:
  # bytes of device memory reserved for the paged KV cache on every replica.  Fixed at deploy time, and
  # smaller than the free memory on a development box so the profiler and the tracer have room.
  kv_pool_bytes: %d
  block_size: 16
  enforce_whole_sequences: true      # a sequence is admitted only if its full context fits
  max_batch_size: %d                 # the batch the replicas run today
traffic:
  context_tokens: %d                 # the production context length, fixed by the product
  slo_p99_ms: %.0f
  growth_next_quarter: %.2f          # applied to every tenant
""" % (int(round(p["N"] / 1e6)), int(p["kv_pool"]), int(p["prod_batch"]), int(p["prod_seq"]),
       p["slo_ms"], GROWTH)
    return {"notebook/runs.jsonl": "\n".join(json.dumps(r, sort_keys=True) for r in ctx["rows"]) + "\n",
            "deploy/serving.yaml": yaml}


# -------------------------------------------------------------------------------------------------- items
def _opt(b):
    """Option labels are strings: `grade_item` looks the submitted choice up in the regret dict, and an
    integer that round-trips through JSON as a string would be scored as an unknown option."""
    return "batch=%d" % int(b)


def _keys(pf):
    seq, B0, rates, menu = int(pf["prod_seq"]), int(pf["prod_batch"]), pf["tenant_rates"], pf["menu"]
    truth = _mix_p99(pf, seq, B0, rates)
    lo = _mix_p99(pf, seq, B0, _rates_next(pf, GROWTH))
    hi = _mix_p99(pf, seq, B0, _rates_next(pf, TOP_HI))
    fit = _b_fit(pf, seq, pf["kv_pool"])
    loss = {}
    for b in menu:
        v = _mix_p99(pf, seq, int(b), rates)
        loss[_opt(b)] = FAIL_MS if (v is None or int(b) > fit) else v
    return {"q1": truth, "lo": lo, "hi": hi, "loss": loss, "fit": fit}


def items(p, ctx, tol=None):
    tol = tol or {}
    pf = SV.full(p)
    k = _keys(pf)
    B0, seq, menu = int(pf["prod_batch"]), int(pf["prod_seq"]), pf["menu"]
    rates_txt = ", ".join("%.2f" % r for r in pf["tenant_rates"])
    prior = _naive_mean_rate(pf, seq, B0, pf["tenant_rates"])     # the memo's own reading
    return [
        Q.point("q1", "The fleet's tail latency on **today's** traffic: the equally-weighted mean, over the "
                      "six tenants, of the p99 end-to-end latency **each tenant** experiences, with every "
                      "replica at `batch=%d` and `seq=%d` and each tenant arriving at its own rate (%s "
                      "req/s).  This is the mean of six per-tenant p99s, not the p99 of the pooled traffic. "
                      "Noise-free expected value." % (B0, seq, rates_txt),
                "ms", k["q1"], tol.get("q1", 0.02 * k["q1"]), floor=0.004 * k["q1"],
                cards=["S4"], obstacles=["O14", "O2"], chain=["service", "queue", "aggregate"],
                prior_key={"lo": prior, "hi": prior}),
        Q.interval("q2", "The same quantity **next quarter**, at the same batch and context: every tenant's "
                         "arrival rate grows by the disclosed %.0f%%, and the largest tenant's rate is the "
                         "known unknown of Section 5.  Give the set of values the aggregate takes over that "
                         "documented band (noise-free expected values)."
                   % (100.0 * (GROWTH - 1.0)),
                   "ms", k["lo"], k["hi"], tol.get("q2", 0.02 * k["hi"]), floor=0.004 * k["hi"],
                   cards=["S4"], obstacles=["O13", "O2"], chain=["service", "queue", "extremes"]),
        Q.decision("q3", "Which of these production batch sizes should the replicas run, on today's six "
                         "tenant rates, at `seq=%d`?  The loss of an option is the same aggregate as q1 "
                         "evaluated at that batch, in ms; an option is **inadmissible** and scores a loss of "
                         "%.0f ms if the reserved KV pool of `deploy/serving.yaml` cannot admit that many "
                         "whole sequences of the production context, or if any tenant's queue fails to clear "
                         "at that batch.  Pick the lowest loss." % (seq, FAIL_MS),
                   [_opt(b) for b in menu], k["loss"], 1.0,
                   cards=["S4", "S5"], obstacles=["O6", "O13"],
                   chain=["pool"]),
    ]


def answers_from(ph, ctx, collapse=False):
    out = json.loads(json.dumps(ph.get("answers", ph)))
    if collapse:
        for a in out.values():
            if isinstance(a, dict) and "lo" in a and a["hi"] > a["lo"]:
                a["lo"] = a["hi"] = 0.5 * (a["lo"] + a["hi"])
    return out


# ------------------------------------------------------------------------------------------------- oracle
def oracle_design(pf):
    """Buy the busiest buckets long, and the menu batches only as long as `util` needs (which is: not at all).

    `util` is exact, so E[S] at a batch costs one minimum-duration row and carries no error into the answer.
    Everything the answer's precision depends on is the variability constant, and a row's information about
    it scales as rho/(1-rho), so the budget goes to the top three buckets."""
    seq, B0, rates, menu = int(pf["prod_seq"]), int(pf["prod_batch"]), pf["tenant_rates"], pf["menu"]
    reqs = [{"svc": "load", "seq": seq, "batch": B0, "rate": rates[i], "dur": OR_DUR, "seed": 300 + 10 * i + k}
            for i in OR_IDX for k in range(OR_REPS)]
    # E[S] at every menu batch, from `util` alone; the lowest tenant rate is stable at every one of them
    reqs += [{"svc": "load", "seq": seq, "batch": int(b), "rate": rates[0], "dur": MENU_DUR, "seed": 400 + j}
             for j, b in enumerate(menu)]
    # the reserved pool's capacity needs the exact per-sequence KV bytes
    reqs.append({"svc": "bench", "seq": seq, "batch": B0, "dur": MENU_DUR, "seed": 500})
    return reqs


def _es_by_batch(rows):
    """E[S] per batch from `util`, which the manual guarantees exact: util = rate * E[S], no noise at all."""
    out = {}
    for r in rows:
        if r.get("svc") != "load" or r.get("status") != "ok":
            continue
        b = int(r["config"]["batch"])
        out.setdefault(b, []).append(float(r["util"]) / float(r["config"]["rate"]))
    return {b: float(np.mean(v)) for b, v in out.items()}


def _est_k(rows, es):
    """The variability constant, by inverse-variance average over the purchased rows.

    Each row gives E[T] twice and independently - `p50_ms` and `p99_ms` are separate lognormal draws on the
    same mean, and p50/p99 = ln2/ln100 exactly - so their geometric mean carries sig_lat/sqrt(2).  With rho
    exact from `util`, k enters `E[T]/E[S] - 1 = k * x` linearly in `x = rho/(1-rho)`: the estimator is a
    weighted average of per-row readings, so its conditioning is 1 by construction.  That is the whole
    design: no fit, no inversion, no extrapolation."""
    num = den = 0.0
    for r in rows:
        if r.get("svc") != "load" or r.get("status") != "ok" or r.get("p99_ms") is None:
            continue
        b = int(r["config"]["batch"])
        if b not in es:
            continue
        rho = float(r["util"])
        x = rho / (1.0 - rho)
        if x <= 1e-9:
            continue
        ET = math.sqrt((float(r["p99_ms"]) / LN100) * (float(r["p50_ms"]) / LN2)) * 1e-3
        k_row = (ET / es[b] - 1.0) / x
        sig = SV.BASE["sig_lat"] / math.sqrt(float(r["config"]["dur"]) / SV.DUR_REF) / math.sqrt(2.0)
        var = (sig * (ET / es[b]) / x) ** 2
        num += k_row / var
        den += 1.0 / var
    return num / den if den > 0 else float("nan")


def _answer(pf, es, k, fit):
    """The three answers from (E[S] per batch, k, the pool's capacity).  Pure arithmetic from here."""
    seq, B0, rates, menu = int(pf["prod_seq"]), int(pf["prod_batch"]), pf["tenant_rates"], pf["menu"]

    def agg(batch, rs):
        S = es[int(batch)]
        acc = 0.0
        for r in rs:
            f = _factor(k, r * S)
            if f is None:
                return None
            acc += f
        return 1e3 * LN100 * S * acc / len(rs)

    q1 = agg(B0, rates)
    lo = agg(B0, _rates_next(pf, GROWTH))
    hi = agg(B0, _rates_next(pf, TOP_HI))
    loss = {}
    for b in menu:
        v = agg(int(b), rates)
        loss[_opt(b)] = FAIL_MS if (v is None or int(b) > fit) else v
    best = min(loss, key=loss.get)
    return {"q1": {"lo": q1, "hi": q1}, "q2": {"lo": lo, "hi": hi}, "q3": {"choice": best}}, loss


def oracle(sess, rows_nb, ctx, rng, drop=None):
    pf = SV.full(sess.p)
    if drop:
        # G9 asks whether a mechanism is load-bearing for the *answers*.  Neutralise it in the world and
        # recompute the keys exactly, so a kill is the mechanism and never a bad draw.
        p2 = SV.full(json.loads(json.dumps(pf)))
        for path, v in drop.items():
            _set(p2, path, v)
        k2 = _keys(p2)
        best = min(k2["loss"], key=k2["loss"].get)
        ans = {"q1": {"lo": k2["q1"], "hi": k2["q1"]}, "q2": {"lo": k2["lo"], "hi": k2["hi"]},
               "q3": {"choice": best}}
        return ans, {"answers": ans, "drop": sorted(drop)}
    rows = run_rows(sess, oracle_design(pf))
    es = _es_by_batch(rows)
    k = _est_k([r for r in rows if int(r["config"]["batch"]) == int(pf["prod_batch"])], es)
    bench = [r for r in rows if r.get("svc") == "bench"]
    per = float(bench[0]["kv_bytes_per_seq"]) if bench else SV.kv_per_token(pf) * float(pf["prod_seq"])
    fit = int(pf["kv_pool"] // per)
    ans, loss = _answer(pf, es, k, fit)
    return ans, {"answers": ans, "n_rows": len(rows), "k_hat": k, "fit": fit,
                 "es_ms": {b: 1e3 * v for b, v in es.items()}, "loss": loss}


def cert_requests(spec_, rng):
    """The known unknown is a scenario constant no service reads, but the random certificate sample spreads
    its draws over five knobs, so the configurations a question actually mentions are made explicit: every
    menu batch at the production context, over a wide span of arrival rates, plus the notebook's own."""
    out = []
    for b in (8, 16, 24, 32, 48, 64, 96, 120):
        for rate in (0.5, 2.0, 8.0, 32.0):
            out.append({"svc": "load", "seq": 2048, "batch": b, "rate": rate, "dur": 20.0, "seed": 11})
        out.append({"svc": "bench", "seq": 2048, "batch": b, "dur": 20.0, "seed": 12})
    return out


# ------------------------------------------------------------------------------------------------- rivals
def rivals(p, rows_nb, ctx, rng):
    pf = SV.full(p)
    seq, B0, rates, menu = int(pf["prod_seq"]), int(pf["prod_batch"]), pf["tenant_rates"], pf["menu"]
    nb = int(pf["nb_batch"])
    key = _keys(pf)
    best = min(key["loss"], key=key["loss"].get)
    out = {}

    def reg(name, q1=None, lo=None, hi=None, choice=None):
        q1 = key["q1"] if q1 is None else q1
        lo = key["lo"] if lo is None else lo
        hi = key["hi"] if hi is None else hi
        out[name] = {"q1": {"lo": q1, "hi": q1}, "q2": {"lo": lo, "hi": hi},
                     "q3": {"choice": best if choice is None else choice}}

    def mix(batch, rs, k=None):
        return _mix_p99(pf, seq, int(batch), rs, k=k)

    def nxt(top, batch=None, k=None):
        return mix(B0 if batch is None else batch, _rates_next(pf, top), k=k)

    # --- q1/q2: which step of the chain was skipped -------------------------------------------------
    # `service`: the notebook's load rows were taken at the development batch, and its E[S] is used for the
    # production one.  Two variants, because the analyst may or may not notice the rates are utilisations.
    sc = _es(pf, seq, B0) / _es(pf, seq, nb)
    reg("skip:service",
        q1=mix(nb, [r * sc for r in rates]),
        lo=mix(nb, [r * sc for r in _rates_next(pf, GROWTH)]),
        hi=mix(nb, [r * sc for r in _rates_next(pf, TOP_HI)]))
    reg("skip:service2", q1=mix(nb, rates), lo=nxt(GROWTH, batch=nb), hi=nxt(TOP_HI, batch=nb))

    # `queue`: the M/M/1 default (k = 1) instead of measuring the variability of the service time - the
    # single most common tail-latency error, and the one S4 exists to punish
    reg("skip:queue", q1=mix(B0, rates, k=1.0), lo=nxt(GROWTH, k=1.0), hi=nxt(TOP_HI, k=1.0))

    # `wait`: no queueing at all - the p99 of the service time itself
    w = 1e3 * LN100 * _es(pf, seq, B0)
    reg("skip:wait", q1=w, lo=w, hi=w)

    # `aggregate`: the Jensen shortcut and the memo's own recommendation - the p99 at the mean arrival rate
    reg("skip:aggregate", q1=_naive_mean_rate(pf, seq, B0, rates),
        lo=_naive_mean_rate(pf, seq, B0, _rates_next(pf, GROWTH)),
        hi=_naive_mean_rate(pf, seq, B0, _rates_next(pf, TOP_HI)))

    # `extremes`: the known unknown collapsed to the midpoint of its band, as commercial proposes
    mid = 0.5 * (GROWTH + TOP_HI)
    reg("skip:extremes", lo=nxt(mid), hi=nxt(mid))

    # `quantile`: p50 read as p99 (the ratio ln2/ln100 dropped)
    reg("skip:quantile", q1=key["q1"] * LN2 / LN100, lo=key["lo"] * LN2 / LN100, hi=key["hi"] * LN2 / LN100)

    # the memo's headline, end to end: the mean rate at the development batch
    bp_ = _naive_mean_rate(pf, seq, nb, rates)
    reg("B_prior", q1=bp_, lo=_naive_mean_rate(pf, seq, nb, _rates_next(pf, GROWTH)),
        hi=_naive_mean_rate(pf, seq, nb, _rates_next(pf, TOP_HI)))

    # --- q3: which constraint was skipped ----------------------------------------------------------
    # `pool`: the development box's `batch_max` used as the deployable ceiling instead of the reserved pool,
    # so the largest menu batch - which benchmarks best of all - is chosen
    big = {}
    for b in menu:
        v = mix(b, rates)
        big[_opt(b)] = FAIL_MS if v is None else v
    reg("skip:pool", choice=min(big, key=big.get))

    # `stability`: the queueing formula applied without checking that the queue clears.  This rival is
    # *informational*, and the reason is a fact about the axis rather than a weakness in the rival.  On
    # servelab, E[S] falls monotonically with `batch`, so every tenant's rho = rate * E[S] falls with it
    # too: the option whose queue fails to clear (menu[0], where the top three tenants sit above rho = 1)
    # is necessarily also the option with the largest service time, hence the worst loss on any scoring
    # that is increasing in E[S].  A stability screen can therefore never be the *binding* constraint of a
    # batch decision - its verdict is already implied by the service term - and no shortcut that skips it
    # can be made to pick a wrong option without also getting the service time wrong, which is what
    # `skip:service` already tests.  Stability stays in q3's *text* because it is what makes menu[0]'s
    # loss defined at all (`_mix_p99` returns None there), but it is not claimed as a derivation step:
    # see the `chain` of q3, which is `["pool"]` alone.  The rival is registered so the matrix records
    # the measurement rather than leaving the question open.
    st = {}
    kk = 0.5 * (1.0 + pf["cs2"])
    for b in menu:
        S = _es(pf, seq, int(b))
        acc = sum(1.0 + kk * (r * S) / (1.0 - r * S) for r in rates)
        v = 1e3 * LN100 * S * acc / len(rates)
        st[_opt(b)] = FAIL_MS if int(b) > key["fit"] else v
    reg("skip:stability", choice=min(st, key=st.get))

    # G12 names a nuisance card and G11 a reasoning step; where the two are the same shortcut seen from
    # different sides the same answers are registered under both names rather than a rival being invented.
    out["naive_ignore:S4"] = out["skip:queue"]
    out["naive_ignore:S5"] = out["skip:pool"]
    return out


def rival_designs(p, rows_nb, rng):
    return {}


# `cs2` = 0 is the deterministic-service-time world: k = 1/2, so the whole queueing term halves.  `kv_bytes`
# is shrunk rather than zeroed because `batch_max` and the pool's capacity both divide by it - at 0.05 the
# reserved pool admits every menu batch, so q3's feasibility screen disappears and its answer flips.
DROP = {"S4": {"cs2": 0.0},
        "S5": {"kv_bytes": 0.05}}
INFO_RIVALS = ("skip:stability",)


# ---------------------------------------------------------------------------------------------- posedness
def _admissible(pf):
    """Thresholds are this task's tolerance arithmetic, not taste.

    `calibrate` sets T = 2.25 * p90(|oracle error|).  The oracle reads `util` exactly and combines two
    latency channels over OR_REPS repetitions at the three busiest buckets at `dur=OR_DUR`, which puts T at
    about 1.2% of q1's truth (measured 1.18-1.20% over twelve draws).  G3 wants q2's band at least 4T wide,
    i.e. about 4.8% of the upper endpoint; the band GROWTH..TOP_HI delivers 15-19T, so the constraint that
    actually binds is the *other* side - the top tenant must not saturate, hence RHO_NEXT_MAX.  The Jensen
    floor W_MIN = 1.20 is what makes q1's nearest rival clear 2T with room: at W_MIN the aggregate shortcut
    is 17% of the truth, about 14T.  G8 wants a strictly positive regret gap on q3, which the menu windows
    deliver by construction (one option unstable, one over the SLO, one best, one inadmissible)."""
    seq, B0 = int(pf["prod_seq"]), int(pf["prod_batch"])
    rates, menu = pf["tenant_rates"], pf["menu"]
    S0 = _es(pf, seq, B0)

    # S1's roofline switch must be inert at every batch the questions mention, or CARDS is a misdescription
    # of which mechanism is load-bearing (see the module docstring).
    if SV.kv_per_token(pf) * float(seq) / pf["BW"] <= 2.0 * pf["N"] / (pf["P_peak"] * pf["eff"]):
        return False, "the roofline switch is not inert: arithmetic can bind at production context"

    truth = _mix_p99(pf, seq, B0, rates)
    naive = _naive_mean_rate(pf, seq, B0, rates)
    if truth is None or naive is None:
        return False, "the production mix does not clear at the production batch"
    if truth / naive < W_MIN:
        return False, "Jensen ratio %.3f below %.2f" % (truth / naive, W_MIN)

    hi = _mix_p99(pf, seq, B0, _rates_next(pf, TOP_HI))
    lo = _mix_p99(pf, seq, B0, _rates_next(pf, GROWTH))
    if hi is None or lo is None:
        return False, "next quarter does not clear at the production batch"
    if (hi - lo) / hi < 0.06:
        return False, "q2's band is %.3f%% of its upper endpoint, too narrow for four tolerances" % (
            100.0 * (hi - lo) / hi)

    key = _keys(pf)
    regs = sorted(v - min(key["loss"].values()) for v in key["loss"].values())
    if regs[1] <= 0.0:
        return False, "q3 has no strictly positive regret gap"
    if key["loss"][_opt(menu[1])] <= pf["slo_ms"]:
        return False, "the batch the fleet runs today already meets the SLO: no decision"
    if key["loss"][_opt(menu[2])] >= pf["slo_ms"]:
        return False, "no menu option meets the SLO"
    if int(menu[3]) <= key["fit"]:
        return False, "the reserved pool admits the largest menu batch: q3's feasibility screen is inert"

    # every oracle row must be inside the per-request cap and the whole design inside the total
    cost = OR_REPS * len(OR_IDX) * OR_DUR + (len(menu) + 1) * MENU_DUR
    caps = spec(pf)["caps"]
    n = OR_REPS * len(OR_IDX) + len(menu) + 1
    if OR_DUR > caps["run_cost"] or cost > caps["total_cost"] or n > caps["max_runs"]:
        return False, "the reference design does not fit its own caps (%d runs, %.0f)" % (n, cost)

    return True, ("Jensen ratio %.3f, q2 band %.2f%% of the upper endpoint, q3 gap %.1f ms, pool admits %d "
                  "of %d, reference design %d runs / %.0f of %.0f"
                  % (truth / naive, 100.0 * (hi - lo) / hi, regs[1], key["fit"], int(menu[3]), n, cost,
                     caps["total_cost"]))


def _stray_collisions(w):
    """Which hidden constants the row-derived exemption is covering for, beyond the ones it argues for.

    `public_values` is a list of *numbers*, not of names, so declaring a measured `util` of 0.8532 exempts
    anything else in the world that renders as 0.8532 - and `mem_util` is drawn in [0.84, 0.92], which is
    exactly where the top of the utilisation ladder lives.  That is a real hole: the argument for the
    exemption is that a disclosed *measurement* does not leak a *constant* (the agent can make the same
    measurement for the price of one row), and it does not extend to an unrelated parameter that happens to
    print the same digits.  So the scan is re-run with the row-derived numbers withdrawn and **any**
    remaining hit rejects the draw - there is no MAY_COLLIDE here, because every number this task prints is
    either a scenario constant it chose or a measurement the agent can repeat."""
    from .. import build as _B
    keep = set(map(id, w["ctx"]["obs"]))
    pub = [v for v in public_values(w["ctx"]) if id(v) not in keep]
    text = w["notes"] + json.dumps(w["rows"]) + "\n".join(w["files"].values())
    return _B.leakage_scan(w, [], text, pub)["hits"]


def wellposed(w):
    ok, why = _admissible(SV.full(w["pf"]))
    if not ok:
        return False, why
    stray = _stray_collisions(w)
    if stray:
        return False, ("a hidden constant is legible in the shipped prose: "
                       + ", ".join("%s as %s" % (h["what"], h["pattern"]) for h in stray[:3]))
    return True, why


def public_values(ctx):
    """Every number the agent is *given*: the scenario constants this task chose and printed, and the
    readings of the notebook's own rows.  `_stray_collisions` withdraws the second group and re-scans."""
    return list(ctx["pub"]) + list(ctx["obs"]) + [
        ctx["p99_nb"], ctx["p50_nb"], ctx["util_nb"], ctx["mean_rate"], ctx["tp_ratio"]]
