"""Estimator for production-mix human win rate.

The free judge verdict is useful, but biased.  We estimate the human win
probability separately for each topic/length/judge-verdict cell, then weight
the cell totals by the production stratum mix.
"""


# Small calibration sample measured in the lab.  Entries are
# (human wins, labels) for P(human win | topic, length, judge verdict).
_CAL = {
    ("code", "long", False): (0, 13),
    ("code", "long", True): (12, 13),
    ("code", "short", False): (1, 13),
    ("code", "short", True): (12, 13),
    ("factual", "long", False): (0, 13),
    ("factual", "long", True): (12, 13),
    ("factual", "short", False): (2, 13),
    ("factual", "short", True): (13, 13),
    ("math", "long", False): (1, 13),
    ("math", "long", True): (11, 13),
    ("math", "short", False): (0, 13),
    ("math", "short", True): (13, 13),
    ("writing", "long", False): (0, 13),
    ("writing", "long", True): (13, 13),
    ("writing", "short", False): (1, 13),
    ("writing", "short", True): (13, 13),
}

_PRIOR_WEIGHT = 1.0
_Z_90_WITH_CUSHION = 1.70


def _mix_weight(env, topic, length):
    return float(env.production_mix.get(topic + "|" + length, 0.0))


def _prior_params(topic, length, judge_win):
    wins, n = _CAL.get((topic, length, judge_win), (1, 2))
    # Jeffreys smoothing plus a discounted lab calibration sample.  The
    # discount keeps the hidden arena's own labels in charge if it differs.
    return 0.5 + _PRIOR_WEIGHT * wins, 0.5 + _PRIOR_WEIGHT * (n - wins)


def _beta_terms(a, b):
    s = a + b
    if s <= 0.0:
        return 0.25, 0.0, 0.5
    mean = a / s
    e_pq = (a * b) / (s * (s + 1.0))
    var_p = (a * b) / (s * s * (s + 1.0))
    return e_pq, var_p, mean


def _cell_variance(cell, extra_labels):
    n_total = cell["N"]
    n_lab = cell["n"] + extra_labels
    if n_lab >= n_total:
        return 0.0
    a, b = cell["a"], cell["b"]

    # Expected posterior after extra labels at the current posterior mean.
    _, _, p = _beta_terms(a, b)
    aa = a + extra_labels * p
    bb = b + extra_labels * (1.0 - p)
    e_pq, var_p, _ = _beta_terms(aa, bb)
    m = n_total - n_lab
    c = cell["coef"]
    return c * c * (m * e_pq + m * m * var_p)


def _spread_sample(ids, k):
    if k <= 0:
        return []
    if k >= len(ids):
        return list(ids)
    out = []
    used = set()
    n = len(ids)
    for i in range(k):
        idx = int((i + 0.5) * n / k)
        if idx >= n:
            idx = n - 1
        while idx in used and idx + 1 < n:
            idx += 1
        while idx in used and idx > 0:
            idx -= 1
        used.add(idx)
        out.append(ids[idx])
    return out


def estimate(env):
    strata_n = {}
    for r in env.rows:
        sk = (r["topic"], r["length"])
        strata_n[sk] = strata_n.get(sk, 0) + 1

    cells = {}
    for r in env.rows:
        topic = r["topic"]
        length = r["length"]
        judge_win = bool(r["judge_win"])
        key = (topic, length, judge_win)
        if key not in cells:
            sn = strata_n[(topic, length)]
            weight = _mix_weight(env, topic, length)
            a, b = _prior_params(topic, length, judge_win)
            cells[key] = {
                "ids": [],
                "N": 0,
                "n": 0,
                "x": 0,
                "a0": a,
                "b0": b,
                "a": a,
                "b": b,
                "coef": weight / float(sn) if sn else 0.0,
            }
        cells[key]["ids"].append(int(r["id"]))
        cells[key]["N"] += 1

    budget = int(env.label_budget)
    alloc = {key: 0 for key in cells}

    # Greedy Neyman-like allocation: repeatedly buy the label with the largest
    # modelled reduction in posterior uncertainty for the production-weighted
    # target.  This automatically focuses on production-heavy strata.
    for _ in range(min(budget, len(env.rows))):
        best_key = None
        best_gain = 0.0
        for key, cell in cells.items():
            if alloc[key] >= cell["N"]:
                continue
            before = _cell_variance(cell, alloc[key])
            after = _cell_variance(cell, alloc[key] + 1)
            gain = before - after
            if gain > best_gain:
                best_gain = gain
                best_key = key
        if best_key is None or best_gain <= 0.0:
            break
        alloc[best_key] += 1

    ids_to_label = []
    id_to_key = {}
    for key, k in alloc.items():
        ids = sorted(cells[key]["ids"])
        chosen = _spread_sample(ids, k)
        ids_to_label.extend(chosen)
        for pid in chosen:
            id_to_key[pid] = key

    if ids_to_label:
        # The public lab caps a label call at 200 ids; use chunks in case the
        # hidden environment mirrors that limit.
        for start in range(0, len(ids_to_label), 200):
            got = env.label(ids_to_label[start:start + 200])
            for sid, val in got.items():
                key = id_to_key[int(sid)]
                cell = cells[key]
                cell["n"] += 1
                if bool(val):
                    cell["x"] += 1

    point = 0.0
    var = 0.0
    for cell in cells.values():
        n_total = cell["N"]
        n_lab = cell["n"]
        x = cell["x"]
        a = cell["a0"] + x
        b = cell["b0"] + n_lab - x
        e_pq, var_p, p_mean = _beta_terms(a, b)
        m = n_total - n_lab
        point += cell["coef"] * (x + m * p_mean)
        var += cell["coef"] * cell["coef"] * (m * e_pq + m * m * var_p)

    if var < 0.0:
        var = 0.0
    half = _Z_90_WITH_CUSHION * (var ** 0.5)
    lo = max(0.0, point - half)
    hi = min(1.0, point + half)
    env.submit(point, lo, hi)
