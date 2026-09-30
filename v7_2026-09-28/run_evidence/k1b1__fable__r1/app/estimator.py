"""Bias-corrected win-rate estimator (stdlib only).

Design: post-stratify every prompt into a cell (topic, length, judge verdict).
Cell sizes are known exactly from the free judge output.  Inside each cell we
buy a uniform random sample of human labels and estimate P(human win | cell).
The stratum win rate is the judge-share-weighted mix of its two cells, and the
production win rate is the production-mix-weighted sum over strata.  The judge
verdict therefore only acts as a (very informative) stratifier - the estimate
is unbiased regardless of how wrong the judge is in any cell.

Labels are allocated in two phases: a proportional pilot, then Neyman
allocation using the pilot's cell standard deviations.

Variance = within-cell sampling variance (with finite population correction)
+ the binomial variance of the arena's finite stratum rate around the
underlying process rate, so the interval is valid whether the target is the
arena's realised rate or the process rate.
"""
import math
import random


def _cells(rows):
    cells = {}
    for r in rows:
        key = (r["topic"] + "|" + r["length"], bool(r["judge_win"]))
        cells.setdefault(key, []).append(r["id"])
    return cells


def _shrunk(k, n):
    """Posterior-ish mean for variance purposes (avoids sd=0 at k in {0, n})."""
    return (k + 1.0) / (n + 2.0)


def _neyman(weights, budget, taken, cap, min_per_cell):
    """Water-filling Neyman allocation with caps. Returns extra labels per cell."""
    extra = {k: 0 for k in weights}
    left = budget
    # minimum per cell
    for k in weights:
        need = max(0, min_per_cell - taken[k])
        need = min(need, cap[k] - taken[k], left)
        extra[k] += need
        left -= need
    # proportional fill with caps
    while left > 0:
        active = [k for k in weights if taken[k] + extra[k] < cap[k] and weights[k] > 0]
        if not active:
            break
        tot = sum(weights[k] for k in active)
        target = {k: weights[k] / tot * left for k in active}
        gave = 0
        for k in active:
            room = cap[k] - taken[k] - extra[k]
            g = min(int(math.floor(target[k])), room)
            extra[k] += g
            gave += g
        if gave == 0:
            # hand out leftovers one at a time by largest fractional need
            order = sorted(active, key=lambda k: -target[k])
            for k in order:
                if left - gave <= 0:
                    break
                if taken[k] + extra[k] < cap[k]:
                    extra[k] += 1
                    gave += 1
            if gave == 0:
                break
        left -= gave
    return extra


def estimate(env):
    rows = env.rows
    mix = env.production_mix
    budget = int(env.label_budget)
    rng = random.Random(12345 + len(rows))

    cells = _cells(rows)
    for ids in cells.values():
        rng.shuffle(ids)
    keys = sorted(cells)
    N = {k: len(cells[k]) for k in keys}
    stratum_n = {}
    for (s, j), n in N.items():
        stratum_n[s] = stratum_n.get(s, 0) + n

    # importance of a cell for the production-mix estimate: w_s * (N_cell / N_s)
    imp = {}
    for (s, j) in keys:
        w = float(mix.get(s, 0.0))
        imp[(s, j)] = w * N[(s, j)] / float(stratum_n[s])

    taken = {k: 0 for k in keys}
    wins = {k: 0 for k in keys}

    def buy(alloc):
        ids = []
        owner = {}
        for k, m in alloc.items():
            for i in cells[k][taken[k]:taken[k] + m]:
                ids.append(i)
                owner[i] = k
        if not ids:
            return
        got = env.label(ids)
        for i in ids:
            k = owner[i]
            taken[k] += 1
            if got.get(str(i), got.get(i, False)):
                wins[k] += 1

    # ---- phase 1: pilot, proportional to importance, min 4 per cell -------
    pilot_budget = int(round(budget * 0.5))
    sd0 = {k: imp[k] * 0.35 for k in keys}
    buy(_neyman(sd0, pilot_budget, taken, N, min_per_cell=4))

    # ---- phase 2: Neyman using pilot standard deviations ------------------
    left = budget - sum(taken.values())
    if left > 0:
        w2 = {}
        for k in keys:
            n = taken[k]
            if n > 0:
                m = _shrunk(wins[k], n)
                sd = math.sqrt(m * (1.0 - m))
            else:
                sd = 0.35
            # Neyman weight ~ importance * sd; blend sd a bit toward the prior
            sd = 0.75 * sd + 0.25 * 0.35
            w2[k] = imp[k] * sd
        buy(_neyman(w2, left, taken, N, min_per_cell=0))

    # ---- estimate ---------------------------------------------------------
    point = 0.0
    var = 0.0
    strata = sorted(stratum_n)
    for s in strata:
        w = float(mix.get(s, 0.0))
        if w <= 0:
            continue
        p_s = 0.0
        v_s = 0.0
        for j in (False, True):
            k = (s, j)
            if k not in N or N[k] == 0:
                continue
            share = N[k] / float(stratum_n[s])
            n = taken[k]
            if n > 0:
                a = wins[k] / float(n)
                m = _shrunk(wins[k], n)
                fpc = max(0.0, 1.0 - n / float(N[k]))
                va = m * (1.0 - m) / n * fpc
            else:
                # never labelled (only if budget was tiny); fall back to judge
                a = 1.0 if j else 0.0
                va = 0.25
            p_s += share * a
            v_s += share * share * va
        # arena's realised stratum rate vs underlying process rate
        ps_shr = min(max(p_s, 0.02), 0.98)
        v_s += ps_shr * (1.0 - ps_shr) / float(stratum_n[s])
        point += w * p_s
        var += w * w * v_s

    se = math.sqrt(max(var, 1e-12))
    z = 1.9  # above the 90% normal quantile: small-cell and process-noise safety
    lo = max(0.0, point - z * se)
    hi = min(1.0, point + z * se)
    env.submit(point, lo, hi)
