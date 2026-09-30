"""Buyable-route algebra and the difficulty certificate B.

v4 measured five intended difficulty levers and four of them did not bind.  Twenty frontier runs passed
18 of 20 instances; declared chain depth, nuisance count, anti-prior gap and budget tightness all failed
to predict a single failure.  The one item that separated two frontier models (e05 q2, 4.1 tolerances low
for gpt-6-astra in both repetitions, 0.1 and 0.6 for fable-5-1) failed for a reason none of those levers
names: the model bought a *slope* and multiplied it, where the answer needed an extrapolation through a
saturating response.  Its own words, from the ledger: "approximately 0.001 proxy units per token at the
proposal's distance".

That is a **route**: a procedure the agent can actually execute with the lab it is given, which returns a
number, and whose number is wrong.  Difficulty is the distance from the truth to the *nearest* such route,
measured in the item's own tolerance:

    B(item) = min over routes  |route(world) - truth(world)| / T

computed in the **noiseless** world, because a route's bias is a property of the estimator and the world,
not of the seed.  B >= 3 says: every shortcut this module knows how to write lands outside the band, so an
answer inside the band is evidence that the agent did the derivation.  B < 3 says the opposite - a shortcut
gets partial or full credit, and the item cannot distinguish a solver from a guesser no matter how deep its
intended chain is.

`e05.wellposed` already computed this quantity by hand, for one item, from a hand-written list (`seps`),
and the list was found to be incomplete by a frontier run.  The point of this module is that the list is
enumerated mechanically from a declaration of the reference estimator, so a blueprint cannot ship with a
hole in it.

--------------------------------------------------------------------------------------------------------
The families

Routes divide into two kinds.  *Structural* routes break the intended derivation and v4 already registers
them by hand, because only the blueprint knows what its own steps are:

  R0  prior        answer the published constant (v4: `B_prior`)
  R5  drop-card    hold a nuisance mechanism neutral (v4: `naive_ignore:<card>`)
  R6  skip-step    omit one declared step of the chain (v4: `skip:<step>`)

*Numerical* routes are the ones a competent agent reaches for when the configuration it is asked about is
not the configuration it can buy.  They depend only on the shape of the response along the instrument axis,
so they can be enumerated from an `Estimand` and nothing else - this module's whole contribution:

  R1  readoff      answer the nearest buyable configuration as if it were the one asked about
  R2  endpoint     answer the most extreme configuration the lab sells
  R3  local_slope  linearise at the operating point with the local derivative
  R4  secant       linearise with the widest secant the lab sells
  R5n two_point    extrapolate from the two nearest grid points (the *good* shortcut; still biased)
  R6n loglinear    fit a power law and extrapolate ("everything is a power law")
  R7  plugin       evaluate at the mean instead of averaging - the Jensen gap
  R8  saturated    assume the response has already reached its asymptote

R3 and R4 are exactly `e05._q2_slope_routes`'s `local` and `span`; this module reproduces both from the
declaration alone, which is the regression test in `tools/test_routes.py`.

--------------------------------------------------------------------------------------------------------
What actually controls B  (measured, `exp/sweep_B.py`, four sweeps on e05 ws=2 q2)

The obvious theory is that B is geometry.  For a smooth response y(t) queried at `target` from a grid ending
at `t0`, the secant route's error is the second-order remainder, so

    |R4 - truth|  ~  |y''| * (target - t0)^2 / 2

which says: push the target out of reach, or raise the curvature, and difficulty follows.  Both knobs were
built and swept, and **both are falsified**.  Scaling e05's length-penalty grid down moved that numerator by
6x (B_floor 5.8 -> 36.3) and B did not rise; it fell, 0.90 -> 0.27.  Halving the saturation scale `sl` at
constant reach did the same, 0.90 -> 0.40, non-monotonically.

The reason is that T is not a constant.  T = max(floor, 2.25 * p90(oracle error)), so the *reference*
estimator's precision sits in the denominator, and e05 q2's reference estimator is a three-parameter tanh
fit extrapolated back to zero distance.  Shrinking the grid or sharpening the bend destroys that fit's
conditioning.  Writing kappa for the amplification from measurement noise sigma to estimator error,

    B  =  bias_geom / (2.25 * kappa * sigma / sqrt(n))

and both geometry knobs raise `bias_geom` by 6x while raising `kappa` by more.  The two remaining factors
were swept and behave exactly as that formula predicts: d log B / d log sigma = **-0.93** (theory -1) and
d log B / d log n_runs = **+0.48** (theory +0.5).

So B is controllable, but not through geometry - through the conditioning of the reference estimator and
through precision bought with runs.  The design rule that follows, and the reason this module exists:

    a good hard item has a WELL-CONDITIONED reference estimator (kappa ~ 1: interpolate, average, compare
    matched groups - never extrapolate) and an O(1) shortcut bias set by an independent world constant.

An item whose own oracle must extrapolate is self-defeating: bias and oracle error are amplified by the same
kappa, so B ~ sqrt(budget) and the floor is unaffordable.  Measured on e05 q2: reaching B = 3 needs ~987 runs
against a 130-run budget.  `exp/kappa_demo.py` builds the matched pair that proves the converse - same lab,
same noise, same 24-run budget, kappa ~ 1 instead of kappa ~ 7-30.

One diagnostic follows for free.  Scale sigma at fixed geometry and fit d log B / d log sigma: about -1 means
noise-limited, so B can be bought with reps; about 0 means remainder-limited, so B is capped and the
*estimand* has to change.  Call it the orthogonality test; `sweep_B.py --knob noise` runs it.
"""
import math

import numpy as np

# Route family -> (tag, one-line meaning).  The tag prefixes every generated route name so the difficulty
# record can be read by family without string surgery.
FAMILIES = {
    "R0": "answer the published constant instead of measuring this world",
    "R1": "answer the nearest buyable configuration as if it were the one asked about",
    "R2": "answer the most extreme configuration the lab sells",
    "R3": "linearise at the operating point using the local derivative",
    "R4": "linearise using the widest secant the lab sells",
    "R5": "extrapolate from the two nearest grid points",
    "R6": "fit a power law across the grid and extrapolate",
    "R7": "evaluate the nonlinear response at the mean instead of averaging it",
    "R8": "assume the response has already reached its asymptote",
}

# Structural families the blueprint registers by hand; listed so `census` can report coverage across both
# kinds without the caller having to know which is which.
STRUCTURAL_PREFIXES = ("B_prior", "naive_ignore:", "skip:", "drop:", "plan:", "B_guess")


class Estimand:
    """One numeric item's reference estimator, declared so the numerical routes can be enumerated.

    A question of the form "what does the world do at `target`, when the lab only sells `grid`" is fully
    described by a scalar response along one instrument axis plus a map into the answer's unit:

      resp(t)      the *noiseless* observable at instrument value t
      grid         the instrument values the lab will actually sell, in any order
      target       the instrument value the question is about; may lie outside `grid`, which is the point
      to_answer    observable value -> the item's answer unit (default: identity)
      truth        the correct answer, computed from the world rather than from `resp`, so that a coding
                   error in `resp` shows up as a route that lands on the truth rather than as a silent pass
      invert       True when the question asks for the *instrument* value that attains a given observable
                   (`target` is then an observable and `to_answer` maps an instrument value); this is the
                   matched-budget shape, where the routes act on the inverse response
      op           the *operating point*: the instrument value of the configuration the question is about,
                   which is where an agent naturally stands when it reaches for a slope.  Defaults to the
                   grid point farthest from the target, which is the right default for "extrapolate back to
                   a neutral limit" items - the run itself carries the largest effect and the limit is at
                   the other end.  e05 q2's measured failure route is the local derivative *at the run's
                   own length*, not at the cheapest buyable point, so both anchors are emitted.

    Everything is evaluated on one world's parameters, so an Estimand is built per world, not per blueprint.
    """

    def __init__(self, resp, grid, target, truth, to_answer=None, invert=False, name="", op=None):
        self.resp = resp
        self.grid = sorted(float(t) for t in grid)
        self.target = float(target)
        self.truth = float(truth)
        self.to_answer = to_answer or (lambda y: float(y))
        self.invert = bool(invert)
        self.name = name
        self._op = None if op is None else float(op)
        if len(self.grid) < 3:
            raise ValueError("an estimand needs at least three buyable instrument values")

    # -------------------------------------------------------------------------------- geometry helpers
    def _y(self, t):
        return float(self.resp(float(t)))

    @property
    def t_near(self):
        """The buyable instrument value closest to the target - where a stop-short route stops."""
        return min(self.grid, key=lambda t: abs(t - self.target))

    @property
    def t_far(self):
        """The buyable value farthest from the target - the "use the most extreme setting" route."""
        return max(self.grid, key=lambda t: abs(t - self.target))

    @property
    def t_op(self):
        """The configuration the question is about, in instrument units."""
        return self.t_far if self._op is None else self._op

    def _two_nearest(self):
        g = sorted(self.grid, key=lambda t: abs(t - self.target))
        return g[0], g[1]

    def _deriv(self, t):
        h = 1e-4 * max(1.0, abs(t))
        return (self._y(t + h) - self._y(t - h)) / (2 * h)


def _safe(fn):
    try:
        v = float(fn())
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
        return None
    return v if math.isfinite(v) else None


def numerical_routes(est):
    """Enumerate the numerical shortcut family for one estimand.  Returns {route_name: answer}.

    A route that cannot be evaluated in this world (a log of a negative number, a division by a zero
    slope) is dropped rather than reported as zero: an inapplicable route is not evidence of difficulty
    in either direction, and `census` counts how many were applicable.
    """
    if est.invert:
        return _inverse_routes(est)
    out = {}
    a, g, tgt = est.to_answer, est.grid, est.target
    t0, t1 = est._two_nearest()
    tf, top = est.t_far, est.t_op

    out["R1:readoff"] = _safe(lambda: a(est._y(t0)))
    out["R2:endpoint"] = _safe(lambda: a(est._y(tf)))
    # Two anchors for the slope, because both are natural and they differ whenever the response is curved:
    # `@op` is the derivative where the agent is standing (the configuration the question names), `@near` is
    # the derivative at the cheapest buyable point.  `@op` is the route a frontier model actually took.
    out["R3:local@op"] = _safe(lambda: a(est._y(top) + est._deriv(top) * (tgt - top)))
    out["R3:local@near"] = _safe(lambda: a(est._y(t0) + est._deriv(t0) * (tgt - t0)))
    out["R4:secant@op"] = _safe(
        lambda: a(est._y(top) + (est._y(g[-1]) - est._y(g[0])) / (g[-1] - g[0]) * (tgt - top)))
    out["R4:secant@near"] = _safe(
        lambda: a(est._y(t0) + (est._y(g[-1]) - est._y(g[0])) / (g[-1] - g[0]) * (tgt - t0)))
    out["R5:two_point"] = _safe(
        lambda: a(est._y(t0) + (est._y(t1) - est._y(t0)) / (t1 - t0) * (tgt - t0)))
    out["R6:loglinear"] = _safe(lambda: a(_loglinear(est)))
    out["R7:plugin"] = _safe(lambda: a(est._y(float(np.mean(g)))))
    out["R7:average"] = _safe(lambda: a(float(np.mean([est._y(t) for t in g]))))
    out["R8:saturated"] = _safe(lambda: a(est._y(g[-1])))
    return {k: v for k, v in out.items() if v is not None}


def _loglinear(est):
    """Fit log y = c + b log t over the grid and evaluate at the target.  The "everything is a power law"
    route: it is the right answer when the response really is a power law, which is why a blueprint whose
    response *is* a power law cannot use this axis to buy difficulty."""
    xs = [t for t in est.grid if t > 0]
    ys = [est._y(t) for t in xs]
    xs = [x for x, y in zip(xs, ys) if y > 0]
    ys = [y for y in ys if y > 0]
    if len(xs) < 3 or est.target <= 0:
        raise ValueError("log-linear route is not applicable on this axis")
    b, c = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(math.exp(c + b * math.log(est.target)))


def _inverse_routes(est):
    """The matched-configuration shape: the question asks which instrument value attains `target`.

    The routes are the same shortcuts read the other way round - invert a linearisation instead of
    evaluating one - and they are what an agent writes when the observable it needs is reported at some
    configurations but not at the one the question is about.
    """
    out = {}
    a, g = est.to_answer, est.grid
    ys = [est._y(t) for t in g]
    tgt = est.target

    def inv_linear(t_a, t_b):
        ya, yb = est._y(t_a), est._y(t_b)
        if abs(yb - ya) < 1e-15:
            raise ValueError("flat response: not invertible on this pair")
        return t_a + (tgt - ya) * (t_b - t_a) / (yb - ya)

    # nearest observable to the one asked about, answered as if it were exact
    j = int(np.argmin([abs(y - tgt) for y in ys]))
    out["R1:readoff"] = _safe(lambda: a(g[j]))
    out["R2:endpoint"] = _safe(lambda: a(g[-1] if tgt > ys[j] else g[0]))
    out["R3:local_slope"] = _safe(lambda: a(g[j] + (tgt - ys[j]) / est._deriv(g[j])))
    out["R4:secant"] = _safe(lambda: a(inv_linear(g[0], g[-1])))
    out["R5:two_point"] = _safe(lambda: a(inv_linear(g[max(j - 1, 0)], g[min(j + 1, len(g) - 1)])))
    out["R6:loglinear"] = _safe(lambda: a(_inv_loglinear(est, tgt)))
    out["R8:saturated"] = _safe(lambda: a(g[-1]))
    return {k: v for k, v in out.items() if v is not None}


def _inv_loglinear(est, tgt):
    xs = [t for t in est.grid if t > 0]
    ys = [est._y(t) for t in xs]
    keep = [(x, y) for x, y in zip(xs, ys) if y > 0]
    if len(keep) < 3 or tgt <= 0:
        raise ValueError("log-linear inverse is not applicable on this axis")
    b, c = np.polyfit(np.log([x for x, _ in keep]), np.log([y for _, y in keep]), 1)
    if abs(b) < 1e-12:
        raise ValueError("flat power law")
    return float(math.exp((math.log(tgt) - c) / b))


# ------------------------------------------------------------------------------------- the certificate
def certify(truth, tol, routes, floor=3.0, reference=()):
    """Difficulty certificate for one numeric item.

    routes: {name: value in the item's answer unit}, from `numerical_routes` plus whatever the blueprint
            registered by hand (its rivals, projected onto this item).
    reference: route names that ARE the intended derivation for this item and must not be scored as
            shortcuts.  Needed because one family's shortcut is another family's answer: for a point
            extrapolation the reference estimator appears nowhere in the enumeration, but for an
            *aggregation* estimand the reference estimator is literally `R7:average` - buy every bucket,
            average the measurements - and scoring it as a rival would report B = 0 for the one item family
            whose B is largest.  The exclusion is narrow on purpose: naming a route here is a claim that
            executing it requires the derivation the item is testing, and it has to be paid for in runs.
    Returns B, the nearest route, the full sorted table, and whether B clears `floor`.

    A route that lands *on* the truth is not excluded: it is the strongest possible evidence that the item
    is cheap, and it drives B to zero, which is exactly the verdict wanted.
    """
    tol = float(tol)
    ref = set(reference)
    tab = []
    for name, v in sorted(routes.items()):
        if v is None or not math.isfinite(float(v)) or name in ref:
            continue
        d = abs(float(v) - float(truth))
        tab.append({"route": name, "value": float(v), "abs_err": d,
                    "over_T": (d / tol) if tol > 0 else float("inf")})
    tab.sort(key=lambda r: r["over_T"])
    B = tab[0]["over_T"] if tab else float("inf")
    return {"B": B, "floor": floor, "pass": bool(B >= floor), "n_routes": len(tab),
            "nearest": tab[0]["route"] if tab else None, "table": tab,
            "reference": sorted(ref)}


def census(routes):
    """Which families are represented, so a blueprint cannot satisfy G18 with one lucky family.

    Returns {"numerical": [...], "structural": [...], "n_families": k}.  G18 asks for at least two distinct
    numerical families and at least one structural family per numeric item: B is a minimum over a set, and
    a minimum over a set of size one is a claim about one guess, not about the space of shortcuts.
    """
    num, stru = set(), set()
    for name in routes:
        if ":" in name and name.split(":")[0] in FAMILIES:
            num.add(name.split(":")[0])
        elif name.startswith(STRUCTURAL_PREFIXES):
            stru.add(name.split(":")[0])
        else:
            stru.add(name)
    return {"numerical": sorted(num), "structural": sorted(stru), "n_families": len(num) + len(stru)}
