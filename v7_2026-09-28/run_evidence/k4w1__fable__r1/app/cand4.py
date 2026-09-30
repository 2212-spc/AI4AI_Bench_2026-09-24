import math

# Topics where the verifier is reliable (high score => correct).
#   correct q05 ~0.39 ; wrong q95 ~0.15
GOOD = {"combinatorics", "number_theory"}
GOOD_THRESH = 0.5
GOOD_CAP = 12

# Topics where the verifier is *inverted* for repeated wrong answers.
#   correct       ~ N(1.00, 0.40)
#   wrong_rep     ~ N(1.70, 0.36)   (variant-specific attractor answers)
#   wrong_single  ~ N(-0.45, 0.72)
BAD_INIT = 5      # one sample per template first
BAD_MAX = 20
W_VAR = 2.0
W_LLR = 1.0


def _lnorm(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - math.log(sd)


def _llr(s):
    lc = _lnorm(s, 1.0, 0.40)
    lw = math.log(0.5 * math.exp(_lnorm(s, 1.7, 0.36)) + 0.5 * math.exp(_lnorm(s, -0.45, 0.72)))
    return lc - lw


def _score_answers(samples):
    groups = {}
    for v, a, s in samples:
        groups.setdefault(a, []).append((v, s))
    out = {}
    for a, lst in groups.items():
        n = len(lst)
        nvar = len(set(v for v, _ in lst))
        mean = sum(s for _, s in lst) / n
        llr = sum(_llr(s) for _, s in lst)
        out[a] = (W_LLR * llr + W_VAR * nvar, n, nvar, mean)
    return out


def _best_and_margin(samples):
    sc = _score_answers(samples)
    ranked = sorted(sc.items(), key=lambda kv: -kv[1][0])
    best = ranked[0]
    margin = best[1][0] - (ranked[1][1][0] if len(ranked) > 1 else -5.0)
    return best[0], best[1], margin


def _confidence(samples):
    """Higher = more confident. Used to prioritise extra samples."""
    best, (sc, n, nvar, mean), margin = _best_and_margin(samples)
    conf = margin
    if nvar >= 3 and mean < 1.4:
        conf += 3.0 + nvar
    elif nvar >= 2 and mean < 1.35:
        conf += 1.0
    return conf


def run(env):
    out = {}
    qs = env.questions
    good = [q for q in qs if q["topic"] in GOOD]
    bad = [q for q in qs if q["topic"] not in GOOD]
    n_bad = len(bad)
    nv = env.n_variants

    # --- reliable-verifier topics: sample until a confidently-correct answer appears ---
    for i, q in enumerate(good):
        qid = q["id"]
        best_a, best_s = None, -1e9
        # keep at least BAD_INIT per remaining bad question in reserve
        reserve = BAD_INIT * n_bad
        avail = env.left() - reserve
        remaining_good = len(good) - i
        cap = max(1, min(GOOD_CAP, avail // max(1, remaining_good) + 2))
        for j in range(cap):
            if env.left() <= 0:
                break
            a, s = env.sample(qid, j % nv)
            if s > best_s:
                best_a, best_s = a, s
            if s >= GOOD_THRESH:
                break
        out[qid] = best_a if best_a is not None else ""

    # --- inverted-verifier topics: initial round, one sample per template ---
    state = {}
    for q in bad:
        qid = q["id"]
        samples = []
        for v in range(min(BAD_INIT, nv)):
            if env.left() <= 0:
                break
            a, s = env.sample(qid, v)
            samples.append((v, a, s))
        state[qid] = samples

    # --- global greedy: repeatedly give one more sample to the least-confident question ---
    conf = {qid: (_confidence(s) if s else -1e9) for qid, s in state.items()}
    active = {qid for qid, s in state.items() if len(s) < BAD_MAX}
    while env.left() > 0 and active:
        qid = min(active, key=lambda k: conf[k])
        samples = state[qid]
        v = len(samples) % nv
        a, s = env.sample(qid, v)
        samples.append((v, a, s))
        conf[qid] = _confidence(samples)
        if len(samples) >= BAD_MAX:
            active.discard(qid)

    for qid, samples in state.items():
        out[qid] = _best_and_margin(samples)[0] if samples else ""

    for q in qs:
        out.setdefault(q["id"], "")
    env.submit(out)
