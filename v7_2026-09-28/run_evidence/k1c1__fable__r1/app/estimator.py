# Estimator for cand-4's production-mix win rate with a biased LLM judge.
#
# Approach: the target is the production-weighted mean of the *human* verdict over
# the rows of this arena.  Each row sits in one of 16 cells (topic x length x judge
# verdict).  Inside a cell the judge is either right or wrong, and the lab showed the
# judge is right ~97% of the time with the errors concentrated in a few cells
# (misses on short math/writing, false positives on long code/math).  We therefore
#   1. buy human labels cell by cell (adaptive Neyman allocation over three phases,
#      so cells that turn out noisy get more labels),
#   2. treat each cell's P(human win | cell) as Beta-distributed with a weak prior
#      taken from the lab measurements (judge error rates are a property of the
#      judge and transfer between arenas; the prior is weak so runtime labels
#      dominate),
#   3. simulate the unlabelled rows from the posterior (beta-binomial, finite
#      population) and report the posterior mean and 5%/95% quantiles.
import random

Z = 1.645
WIDEN = 1.35            # safety factor on the posterior interval

# Lab measurements: (topic, length, judge_win) -> (human wins, labels), 208 labels
# over two dev arenas.
LAB = {
    ("code", "long", False): (0, 13), ("code", "long", True): (12, 13),
    ("code", "short", False): (0, 13), ("code", "short", True): (13, 13),
    ("factual", "long", False): (0, 13), ("factual", "long", True): (13, 13),
    ("factual", "short", False): (0, 13), ("factual", "short", True): (12, 13),
    ("math", "long", False): (0, 13), ("math", "long", True): (10, 13),
    ("math", "short", False): (2, 13), ("math", "short", True): (13, 13),
    ("writing", "long", False): (0, 13), ("writing", "long", True): (13, 13),
    ("writing", "short", False): (2, 13), ("writing", "short", True): (13, 13),
}
PRIOR_WEIGHT = 5.0      # pseudo-observations carried by the lab prior
POOL_TRUE, POOL_FALSE = 0.97, 0.03   # pooled P(human win | verdict) from the lab
MIN_PQ = 0.03           # variance floor used only for label allocation


def _prior(cell):
    x, n = LAB.get(cell, (0, 0))
    g = POOL_TRUE if cell[2] else POOL_FALSE
    m = (x + 2.0 * g) / (n + 2.0)
    return PRIOR_WEIGHT * m, PRIOR_WEIGHT * (1.0 - m)


class Cell:
    def __init__(self, key, ids, coef):
        self.key = key
        self.ids = list(ids)
        self.coef = coef            # w_s / N_s : weight of one row of this cell
        self.a0, self.b0 = _prior(key)
        self.x = 0                  # human wins among labelled rows
        self.n = 0                  # labelled rows
        self.pending = []

    @property
    def unlabelled(self):
        return len(self.ids) - self.n

    def post(self):
        return self.a0 + self.x, self.b0 + self.n - self.x

    def var_contrib(self, extra=0):
        # approx. variance of this cell's contribution if `extra` more labels are bought
        a, b = self.post()
        m = a / (a + b)
        pq = max(m * (1.0 - m), MIN_PQ)
        u = self.unlabelled - extra
        if u <= 0:
            return 0.0
        k = a + b + extra
        return (self.coef ** 2) * (u * pq + u * (u - 1) * pq / (k + 1.0))


def _allocate(cells, budget):
    # greedy: repeatedly give a label to the cell with the largest marginal variance drop
    plan = {c.key: 0 for c in cells}
    for _ in range(budget):
        best, gain = None, 0.0
        for c in cells:
            if plan[c.key] >= c.unlabelled:
                continue
            g = c.var_contrib(plan[c.key]) - c.var_contrib(plan[c.key] + 1)
            if g > gain:
                best, gain = c, g
        if best is None:
            break
        plan[best.key] += 1
    return plan


def _buy(env, cells, plan, rng):
    ids, owner = [], {}
    for c in cells:
        k = plan[c.key]
        if k <= 0:
            continue
        pool = c.ids[c.n:]
        rng.shuffle(pool)
        c.ids[c.n:] = pool
        chosen = pool[:k]
        ids += chosen
        for i in chosen:
            owner[i] = c
    if not ids:
        return
    got = env.label(ids)
    for i in ids:
        v = got.get(str(i), got.get(i))
        c = owner[i]
        c.n += 1
        c.x += 1 if v else 0


def estimate(env):
    rng = random.Random(12345)
    rows = env.rows
    mix = env.production_mix
    by_stratum = {}
    for r in rows:
        by_stratum.setdefault((r["topic"], r["length"]), []).append(r)

    cells = []
    # renormalise over strata that actually appear in this arena
    wsum = sum(float(mix.get("%s|%s" % st, 0.0)) for st in by_stratum) or 1.0
    for (t, l), rs in by_stratum.items():
        w = float(mix.get("%s|%s" % (t, l), 0.0)) / wsum
        coef = w / len(rs)
        for v in (False, True):
            ids = [r["id"] for r in rs if bool(r["judge_win"]) == v]
            if ids:
                cells.append(Cell((t, l, v), ids, coef))

    budget = int(env.label_budget)
    spent = 0

    # phase 1: a few labels in every cell that carries any production weight
    phase1 = {}
    for c in cells:
        k = 4 if c.coef > 0 else 0
        phase1[c.key] = min(k, c.unlabelled)
    tot = sum(phase1.values())
    if tot > budget:
        for k in phase1:
            phase1[k] = 0
        tot = 0
    _buy(env, cells, phase1, rng)
    spent += tot

    # phases 2 and 3: adaptive Neyman-style allocation of the remainder
    for frac in (0.5, 1.0):
        left = budget - spent
        chunk = int(round(left * frac)) if frac < 1 else left
        if chunk <= 0:
            continue
        plan = _allocate(cells, chunk)
        _buy(env, cells, plan, rng)
        spent += sum(plan.values())

    # posterior simulation of the finite-population production-mix win rate
    fixed = sum(c.coef * c.x for c in cells)
    draws = []
    n_draws = 4000
    for _ in range(n_draws):
        tot = fixed
        for c in cells:
            u = c.unlabelled
            if u == 0 or c.coef == 0:
                continue
            a, b = c.post()
            p = rng.betavariate(a, b)
            # binomial(u, p) via sum of Bernoullis (u is at most a few hundred)
            s = 0
            for _i in range(u):
                if rng.random() < p:
                    s += 1
            tot += c.coef * s
        draws.append(tot)
    draws.sort()
    point = sum(draws) / len(draws)
    lo = draws[int(0.05 * n_draws)]
    hi = draws[min(n_draws - 1, int(0.95 * n_draws))]
    # guard: never narrower than a normal interval from the posterior sd
    var = sum((d - point) ** 2 for d in draws) / (n_draws - 1)
    sd = var ** 0.5
    lo = min(lo, point - Z * sd)
    hi = max(hi, point + Z * sd)
    # the lab prior may be mildly off on a hidden arena; widen to absorb that
    lo = point - WIDEN * (point - lo)
    hi = point + WIDEN * (hi - point)
    lo = max(0.0, lo)
    hi = min(1.0, hi)
    env.submit(point, lo, hi)
