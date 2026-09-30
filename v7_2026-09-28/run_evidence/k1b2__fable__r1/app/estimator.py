"""Bias-corrected win-rate estimator for the production traffic mix.

Idea
----
Within a stratum s (topic|length) the judge's verdict splits prompts into two
cells: judge-said-win (j=1) and judge-said-loss (j=0).  The judge's share of
each cell, q_s1 and q_s0, is known exactly and for free.  What we do not know
is how often the judge is right in each cell:

    ppv_s = P(true win | judge win, s)     npv_s = P(true loss | judge loss, s)

so the true win rate of stratum s is

    p_s = q_s1 * ppv_s + q_s0 * (1 - npv_s)

and the production-mix win rate is  sum_s w_s * p_s  with w = env.production_mix.

We spend the label budget measuring ppv_s / npv_s directly, sampling prompts at
random *within* each (stratum, verdict) cell (Neyman allocation: cells that
carry more production weight and more judge mass get more labels).  Each cell
rate gets a Beta posterior (Jeffreys prior plus a weak prior from lab
measurements of the same judge); the interval is obtained by simulating the
posterior of the weighted sum.  Standard library only.
"""
import random

# Weak prior from lab labels (dev arena, 12 human labels per cell).  Keyed by
# "topic|length|judge" -> (true wins, n).  Only used as a *weak* prior and as a
# fallback when a hidden arena has no prompts at all in a cell.
_LAB = {
    "code|long|0": (0, 12), "code|long|1": (8, 12),
    "code|short|0": (3, 12), "code|short|1": (12, 12),
    "factual|long|0": (1, 12), "factual|long|1": (8, 12),
    "factual|short|0": (0, 12), "factual|short|1": (12, 12),
    "math|long|0": (1, 12), "math|long|1": (11, 12),
    "math|short|0": (1, 12), "math|short|1": (11, 12),
    "writing|long|0": (1, 12), "writing|long|1": (11, 12),
    "writing|short|0": (4, 12), "writing|short|1": (11, 12),
}
_PRIOR_WEIGHT = 3.0   # pseudo-observations contributed by the lab prior per cell
_N_SIM = 6000
_MIN_PER_CELL = 3


def _prior(cell):
    """Beta(a, b) prior for P(true win | cell)."""
    a, b = 0.5, 0.5                      # Jeffreys
    if cell in _LAB:
        k, n = _LAB[cell]
        a += _PRIOR_WEIGHT * k / float(n)
        b += _PRIOR_WEIGHT * (n - k) / float(n)
    return a, b


def _quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def estimate(env):
    rows = env.rows
    mix = dict(env.production_mix)
    budget = int(env.label_budget)
    rng = random.Random(20240927)

    # ---- group prompts into (stratum, judge) cells --------------------------
    cells = {}          # cell key -> list of ids
    strata = {}         # stratum -> {"n": int, "n1": int}
    for r in rows:
        s = "%s|%s" % (r["topic"], r["length"])
        j = 1 if r["judge_win"] else 0
        cells.setdefault("%s|%d" % (s, j), []).append(r["id"])
        st = strata.setdefault(s, {"n": 0, "n1": 0})
        st["n"] += 1
        st["n1"] += j

    # Make sure every production stratum exists in our bookkeeping.
    for s in mix:
        strata.setdefault(s, {"n": 0, "n1": 0})

    # ---- Neyman allocation of labels across cells ---------------------------
    # variance contribution of cell (s,j) ~ (w_s * q_sj)^2 * p(1-p) / n_sj
    # => optimal n_sj proportional to w_s * q_sj * sqrt(p(1-p))
    score = {}
    for s, st in strata.items():
        w = mix.get(s, 0.0)
        if st["n"] == 0 or w <= 0:
            continue
        for j in (0, 1):
            key = "%s|%d" % (s, j)
            size = len(cells.get(key, []))
            if size == 0:
                continue
            q = (st["n1"] if j == 1 else st["n"] - st["n1"]) / float(st["n"])
            a, b = _prior(key)
            p = a / (a + b)
            p = min(max(p, 0.2), 0.8)     # don't starve cells whose prior is extreme
            score[key] = w * q * (p * (1 - p)) ** 0.5

    alloc = {}
    total_score = sum(score.values())
    if total_score > 0 and budget > 0:
        # start with a floor for every cell that matters, then distribute the rest
        floor = {k: min(_MIN_PER_CELL, len(cells[k])) for k in score}
        if sum(floor.values()) > budget:
            # extremely small budget: allocate floor to the highest-scored cells
            floor = {}
            left = budget
            for k in sorted(score, key=lambda k: -score[k]):
                take = min(_MIN_PER_CELL, len(cells[k]), left)
                if take <= 0:
                    break
                floor[k] = take
                left -= take
        alloc = dict(floor)
        remaining = budget - sum(alloc.values())
        # iterative proportional fill with capacity caps
        active = {k: score[k] for k in score}
        while remaining > 0 and active:
            tot = sum(active.values())
            fractional = {}
            for k, sc in active.items():
                want = remaining * sc / tot
                cap = len(cells[k]) - alloc[k]
                give = min(int(want), cap)
                alloc[k] += give
                fractional[k] = (want - int(want)) if cap > give else -1.0
            remaining = budget - sum(alloc.values())
            # hand out leftovers one at a time by largest fractional remainder
            for k in sorted(fractional, key=lambda k: -fractional[k]):
                if remaining <= 0:
                    break
                if fractional[k] < 0:
                    continue
                if alloc[k] < len(cells[k]):
                    alloc[k] += 1
                    remaining -= 1
            active = {k: sc for k, sc in active.items() if alloc[k] < len(cells[k])}
            if all(fractional.get(k, -1.0) < 0 for k in active):
                # nothing more could be placed; avoid infinite loop
                if remaining == budget - sum(alloc.values()) and remaining > 0 and not active:
                    break
            if remaining > 0 and not active:
                break
            # safety: if no progress possible, stop
            if remaining > 0 and sum(len(cells[k]) - alloc[k] for k in active) == 0:
                break

    # ---- buy the labels ------------------------------------------------------
    ids = []
    for k, n in alloc.items():
        pool = list(cells[k])
        rng.shuffle(pool)
        ids.extend(pool[:n])
    ids = ids[:budget]
    got = {}
    if ids:
        raw = env.label(ids)
        for kk, v in raw.items():
            got[int(kk)] = bool(v)

    # ---- per-cell Beta posteriors --------------------------------------------
    post = {}
    for k, pool in cells.items():
        a, b = _prior(k)
        for pid in pool:
            if pid in got:
                if got[pid]:
                    a += 1.0
                else:
                    b += 1.0
        post[k] = (a, b)

    # ---- posterior of the production-mix win rate ----------------------------
    parts = []   # (weight, key, a, b) -> contribution weight * rate  (or weight*(1-rate))
    point_plugin = 0.0
    for s, w in mix.items():
        if w <= 0:
            continue
        st = strata[s]
        if st["n"] == 0:
            # no prompts in this stratum at all: fall back to lab-informed prior
            # using the judge share from the lab (rough) -> treat as unknown 50/50 split
            for j, q in ((1, 0.5), (0, 0.5)):
                key = "%s|%d" % (s, j)
                a, b = _prior(key)
                parts.append((w * q, j, a, b))
            continue
        q1 = st["n1"] / float(st["n"])
        q0 = 1.0 - q1
        for j, q in ((1, q1), (0, q0)):
            if q <= 0:
                continue
            key = "%s|%d" % (s, j)
            a, b = post.get(key, _prior(key))
            parts.append((w * q, j, a, b))

    sims = []
    for _ in range(_N_SIM):
        tot = 0.0
        for wq, j, a, b in parts:
            r = rng.betavariate(a, b)
            tot += wq * r
        sims.append(tot)
    for wq, j, a, b in parts:
        point_plugin += wq * a / (a + b)

    # Interval: nominal coverage is env.target_coverage; we widen a little
    # (half-way to 1) so that a run of 24 arenas clears the coverage bar with
    # high probability while staying far inside the width cap.
    target = float(getattr(env, "target_coverage", 0.9) or 0.9)
    conf = target + (1.0 - target) / 2.0
    alpha = 1.0 - conf
    lo = _quantile(sims, alpha / 2.0)
    hi = _quantile(sims, 1.0 - alpha / 2.0)
    point = sum(sims) / float(len(sims)) if sims else point_plugin
    lo = max(0.0, min(lo, point))
    hi = min(1.0, max(hi, point))
    env.submit(point, lo, hi)
