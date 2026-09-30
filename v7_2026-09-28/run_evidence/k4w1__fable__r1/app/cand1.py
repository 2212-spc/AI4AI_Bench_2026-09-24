import math

# Topics where the verifier is reliable (high score => correct).
GOOD = {"combinatorics", "number_theory"}
GOOD_THRESH = 0.3      # correct q05 ~0.3, wrong q95 ~0.15
GOOD_CAP = 8

# Topics where the verifier is inverted for repeated wrong answers.
# per-sample score distributions (from dev):
#   correct       ~ N(1.00, 0.40)
#   wrong_rep     ~ N(1.70, 0.36)   (variant-specific attractor answers)
#   wrong_single  ~ N(-0.45, 0.72)
BAD_CAP = 12


def _lnorm(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - math.log(sd)


def _llr(s):
    lc = _lnorm(s, 1.0, 0.40)
    lw = max(_lnorm(s, 1.7, 0.36) + math.log(0.5), _lnorm(s, -0.45, 0.72) + math.log(0.5))
    lw = math.log(0.5 * math.exp(_lnorm(s, 1.7, 0.36)) + 0.5 * math.exp(_lnorm(s, -0.45, 0.72)))
    return lc - lw


def _score_answers(samples):
    """samples: list of (variant, answer, score). Returns dict answer -> (score, n, nvar, mean)."""
    groups = {}
    for v, a, s in samples:
        groups.setdefault(a, []).append((v, s))
    out = {}
    for a, lst in groups.items():
        n = len(lst)
        nvar = len(set(v for v, _ in lst))
        mean = sum(s for _, s in lst) / n
        llr = sum(_llr(s) for _, s in lst)
        # cross-variant agreement is the strongest signal of correctness
        sc = llr + 1.5 * nvar + 0.5 * n
        out[a] = (sc, n, nvar, mean)
    return out


def _pick(samples):
    sc = _score_answers(samples)
    return max(sc, key=lambda a: sc[a][0])


def run(env):
    out = {}
    qs = env.questions
    good = [q for q in qs if q["topic"] in GOOD]
    bad = [q for q in qs if q["topic"] not in GOOD]
    n_bad = len(bad)

    # --- reliable-verifier topics: sample until a confidently-correct answer appears ---
    for i, q in enumerate(good):
        qid = q["id"]
        best_a, best_s = None, -1e9
        remaining_q = (len(good) - i) + n_bad
        cap = max(1, min(GOOD_CAP, env.left() // max(1, remaining_q) + 2))
        for j in range(cap):
            if env.left() <= 0:
                break
            a, s = env.sample(qid, j % env.n_variants)
            if s > best_s:
                best_a, best_s = a, s
            if s >= GOOD_THRESH:
                break
        out[qid] = best_a if best_a is not None else ""

    # --- inverted-verifier topics: cross-variant agreement + score band ---
    for i, q in enumerate(bad):
        qid = q["id"]
        remaining_q = n_bad - i
        cap = max(1, min(BAD_CAP, env.left() // max(1, remaining_q)))
        samples = []
        v = 0
        # initial round: one per variant
        while len(samples) < min(cap, env.n_variants) and env.left() > 0:
            a, s = env.sample(qid, v % env.n_variants)
            samples.append((v % env.n_variants, a, s))
            v += 1
        # adaptive extra rounds
        while len(samples) < cap and env.left() > 0:
            sc = _score_answers(samples)
            best = max(sc, key=lambda a: sc[a][0])
            _, n, nvar, mean = sc[best]
            if nvar >= 3 and mean < 1.35:
                break
            if nvar >= 2 and mean < 1.25 and len(samples) >= 7:
                break
            a, s = env.sample(qid, v % env.n_variants)
            samples.append((v % env.n_variants, a, s))
            v += 1
        out[qid] = _pick(samples) if samples else ""

    for q in qs:
        out.setdefault(q["id"], "")
    env.submit(out)
