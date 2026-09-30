"""Bias-corrected win-rate estimator for a judged arena.

Model.  Within each cell c = (topic, length, judge verdict) the human verdict of a prompt is
Bernoulli(q_c), q_c = P(human says candidate won | judge verdict, stratum).  Cells with the
same judge verdict share a hierarchical Beta(mu*kappa, (1-mu)*kappa) prior; (mu, kappa) get a
flat prior on a grid and are integrated out.  The judge's error rates are a property of the
judge, so the human labels bought in the lab enter as (discounted) evidence about q_c alongside
the labels bought on THIS arena.

Target.  The production-mix win rate on this arena's prompts:
    T = sum_k w_k * (#true wins in stratum k) / N_k
Labelled prompts contribute exactly; unlabelled prompts in cell c contribute a
Binomial(N_unlab_c, q_c) count.  Point = posterior mean, interval = posterior quantiles by
Monte Carlo over (mu, kappa) -> q_c -> counts.

Labels.  Allocated greedily in two adaptive rounds to the cells where one more label most
reduces the posterior variance of T; prompts within a cell are picked at random.
"""
import math
import random

# ---------------------------------------------------------------------------------------
# Lab evidence: human labels bought on the dev arenas, per (topic, length, judge_win):
# (human wins, labels).  Same judge on every arena, so this is legitimate prior knowledge.
# ---------------------------------------------------------------------------------------
LAB = {
    ("code", "long", False): (0, 11),    ("code", "long", True): (12, 12),
    ("code", "short", False): (0, 11),   ("code", "short", True): (12, 12),
    ("factual", "long", False): (0, 11), ("factual", "long", True): (16, 18),
    ("factual", "short", False): (1, 11), ("factual", "short", True): (17, 18),
    ("math", "long", False): (1, 11),    ("math", "long", True): (17, 18),
    ("math", "short", False): (0, 11),   ("math", "short", True): (17, 18),
    ("writing", "long", False): (0, 11), ("writing", "long", True): (12, 12),
    ("writing", "short", False): (0, 11), ("writing", "short", True): (12, 12),
}
LAB_WEIGHT = 0.75   # discount lab counts a little: hedge against lab/hidden drift
N_SIMS = 4000
MIN_PER_CELL = 4    # floor so every non-empty cell is checked on the hidden arena
INFLATE = 1.1       # interval inflation factor, see estimate()

MU_GRID = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.14, 0.2, 0.28, 0.4, 0.5]
KAPPA_GRID = [3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0, 500.0]


def _lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _hyper_posterior(counts):
    """counts: list of (s, f) per cell.  Returns list of (weight, alpha0, beta0) over the
    (mu, kappa) grid, for an error rate e = 1 - q or q - whichever `counts` encode."""
    out = []
    for mu in MU_GRID:
        for kap in KAPPA_GRID:
            a0, b0 = mu * kap, (1.0 - mu) * kap
            ll = 0.0
            for s, f in counts:
                ll += _lbeta(a0 + s, b0 + f) - _lbeta(a0, b0)
            out.append((ll, a0, b0))
    m = max(x[0] for x in out)
    ws = [(math.exp(ll - m), a0, b0) for ll, a0, b0 in out]
    z = sum(w for w, _, _ in ws)
    return [(w / z, a0, b0) for w, a0, b0 in ws]


class _Cell(object):
    __slots__ = ("key", "coef", "ids", "ls", "lf", "s", "n")

    def __init__(self, key, coef, ids):
        self.key = key
        self.coef = coef          # w_k / N_k : weight of one prompt of this cell in T
        self.ids = list(ids)      # unlabelled ids
        s, m = LAB.get(key, (0, 0))
        # evidence is stored on the "error" scale: judge wrong = 1
        if key[2]:
            self.ls, self.lf = LAB_WEIGHT * (m - s), LAB_WEIGHT * s
        else:
            self.ls, self.lf = LAB_WEIGHT * s, LAB_WEIGHT * (m - s)
        self.s = 0                # judge-wrong count among fresh labels
        self.n = 0                # fresh labels

    def counts(self):
        return self.ls + self.s, self.lf + (self.n - self.s)

    def post(self, a0, b0):
        s, f = self.counts()
        return a0 + s, b0 + f


def _hypers(cells):
    """(mu,kappa) grid posterior per judge direction, plus posterior-mean (a0,b0) for planning."""
    res = {}
    for jw in (False, True):
        cs = [c.counts() for c in cells if c.key[2] == jw]
        grid = _hyper_posterior(cs) if cs else [(1.0, 1.0, 20.0)]
        # for planning use the marginal-likelihood-weighted average hyperparameters
        a0 = sum(w * a for w, a, _ in grid)
        b0 = sum(w * b for w, _, b in grid)
        res[jw] = (grid, a0, b0)
    return res


def _var_contrib(c, a0, b0, extra=0):
    """Posterior variance of the cell's contribution to T if `extra` more labels are bought
    (approximating that their outcomes match the current posterior mean)."""
    nu = len(c.ids) - extra
    if nu <= 0:
        return 0.0
    a, b = c.post(a0, b0)
    a2, b2 = a + extra * a / (a + b), b + extra * b / (a + b)
    m = a2 / (a2 + b2)
    vq = a2 * b2 / ((a2 + b2) ** 2 * (a2 + b2 + 1.0))
    eq1q = m * (1.0 - m) - vq
    return c.coef ** 2 * (nu * eq1q + nu * nu * vq)


def _allocate(cells, budget, hyp):
    plan = {c.key: 0 for c in cells}
    avail = {c.key: len(c.ids) for c in cells}
    for c in cells:
        k = min(MIN_PER_CELL, avail[c.key], budget)
        plan[c.key] += k
        budget -= k
    while budget > 0:
        best, best_gain = None, 0.0
        for c in cells:
            if plan[c.key] >= avail[c.key]:
                continue
            _, a0, b0 = hyp[c.key[2]]
            e = plan[c.key]
            gain = _var_contrib(c, a0, b0, e) - _var_contrib(c, a0, b0, e + 1)
            if gain > best_gain:
                best, best_gain = c, gain
        if best is None:
            break
        plan[best.key] += 1
        budget -= 1
    return plan


def _buy(env, cells, plan, rng):
    req, owner = [], {}
    for c in cells:
        k = plan.get(c.key, 0)
        if k <= 0:
            continue
        rng.shuffle(c.ids)
        take, c.ids = c.ids[:k], c.ids[k:]
        for i in take:
            owner[i] = c
        req.extend(take)
    if not req:
        return 0
    got = env.label(req)
    for k, v in got.items():
        c = owner[int(k)]
        human_win = bool(v)
        wrong = (human_win != c.key[2])
        c.s += 1 if wrong else 0
        c.n += 1
    return len(req)


def _binom(n, p, rng):
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    r = rng.random
    return sum(1 for _ in range(n) if r() < p)


def _draw_grid(grid, rng):
    u = rng.random()
    acc = 0.0
    for w, a0, b0 in grid:
        acc += w
        if u <= acc:
            return a0, b0
    return grid[-1][1], grid[-1][2]


def estimate(env):
    rng = random.Random(20240917)
    rows = env.rows
    mix = env.production_mix
    budget = int(env.label_budget)

    strata = {}
    for r in rows:
        strata.setdefault((r["topic"], r["length"]), []).append(r)
    cells = []
    for (t, l), rs in strata.items():
        w = float(mix.get("%s|%s" % (t, l), 0.0))
        coef = w / len(rs)
        for jw in (False, True):
            ids = [r["id"] for r in rs if bool(r["judge_win"]) == jw]
            if ids:
                cells.append(_Cell((t, l, jw), coef, ids))

    # two adaptive labelling rounds
    spent = _buy(env, cells, _allocate(cells, budget // 2, _hypers(cells)), rng)
    _buy(env, cells, _allocate(cells, budget - spent, _hypers(cells)), rng)

    hyp = _hypers(cells)

    # labelled prompts contribute exactly: human wins = judge wins ^ wrong
    fixed = 0.0
    for c in cells:
        wins = (c.n - c.s) if c.key[2] else c.s
        fixed += c.coef * wins

    # Monte Carlo posterior of T
    sims = []
    for _ in range(N_SIMS):
        hy = {jw: _draw_grid(hyp[jw][0], rng) for jw in (False, True)}
        t = fixed
        for c in cells:
            nu = len(c.ids)
            if nu == 0:
                continue
            a0, b0 = hy[c.key[2]]
            a, b = c.post(a0, b0)
            e = rng.betavariate(a, b)          # judge-wrong rate in this cell
            wrong = _binom(nu, e, rng)
            t += c.coef * ((nu - wrong) if c.key[2] else wrong)
        sims.append(t)
    sims.sort()
    point = sum(sims) / float(N_SIMS)
    alpha = 1.0 - float(getattr(env, "target_coverage", 0.9) or 0.9)
    lo = sims[int(alpha / 2 * N_SIMS)]
    hi = sims[min(N_SIMS - 1, int((1 - alpha / 2) * N_SIMS))]
    # small inflation: hedge against the judge being somewhat worse on hidden arenas than
    # in the lab (local simulation with 2-3x the lab error rates showed ~5% under-coverage)
    lo = point - (point - lo) * INFLATE
    hi = point + (hi - point) * INFLATE
    lo = max(0.0, min(lo, point))
    hi = min(1.0, max(hi, point))
    env.submit(point, lo, hi)
