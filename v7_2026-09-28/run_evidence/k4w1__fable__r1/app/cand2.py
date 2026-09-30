import math

# Topics where the verifier is reliable (high score => correct).
#   correct   ~ q05 0.39 ; wrong q95 ~0.15  (either repeated or singleton)
GOOD = {"combinatorics", "number_theory"}
GOOD_THRESH = 0.3
GOOD_CAP = 8

# Topics where the verifier is *inverted* for repeated wrong answers.
#   correct       ~ N(1.00, 0.40)
#   wrong_rep     ~ N(1.70, 0.36)   (variant-specific attractor answers)
#   wrong_single  ~ N(-0.45, 0.72)
BAD_CAP = 15
W_VAR = 2.0     # weight per distinct variant that produced the answer
W_N = 0.0       # weight per sample
W_LLR = 1.0     # weight on verifier log-likelihood ratio


def _lnorm(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - math.log(sd)


def _llr(s):
    lc = _lnorm(s, 1.0, 0.40)
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
        out[a] = (W_LLR * llr + W_VAR * nvar + W_N * n, n, nvar, mean)
    return out


def _best_and_margin(samples):
    sc = _score_answers(samples)
    ranked = sorted(sc.items(), key=lambda kv: -kv[1][0])
    best = ranked[0]
    margin = best[1][0] - (ranked[1][1][0] if len(ranked) > 1 else -5.0)
    return best[0], best[1], margin


def _stop(n, nvar, mean, k):
    return (nvar >= 4 and mean < 1.4) or (nvar >= 3 and mean < 1.3 and k >= 7)


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
        remaining_q = (len(good) - i) + n_bad
        cap = max(1, min(GOOD_CAP, env.left() // max(1, remaining_q) + 2))
        for j in range(cap):
            if env.left() <= 0:
                break
            a, s = env.sample(qid, j % nv)
            if s > best_s:
                best_a, best_s = a, s
            if s >= GOOD_THRESH:
                break
        out[qid] = best_a if best_a is not None else ""

    # --- inverted-verifier topics: cross-variant agreement + score band ---
    state = {}   # qid -> samples list
    for i, q in enumerate(bad):
        qid = q["id"]
        remaining_q = n_bad - i
        cap = max(1, min(BAD_CAP, env.left() // max(1, remaining_q)))
        samples = []
        v = 0
        while len(samples) < min(cap, nv) and env.left() > 0:
            a, s = env.sample(qid, v % nv)
            samples.append((v % nv, a, s))
            v += 1
        while len(samples) < cap and env.left() > 0:
            best, (_, n, nvar_, mean), _m = _best_and_margin(samples)
            if _stop(n, nvar_, mean, len(samples)):
                break
            a, s = env.sample(qid, v % nv)
            samples.append((v % nv, a, s))
            v += 1
        state[qid] = samples

    # --- final pass: spend any leftover budget on the least-confident bad-topic questions ---
    if bad:
        guard = 0
        while env.left() > 0 and guard < 10000:
            guard += 1
            cand = []
            for qid, samples in state.items():
                if not samples or len(samples) >= 20:
                    continue
                best, (_, n, nvar_, mean), margin = _best_and_margin(samples)
                if _stop(n, nvar_, mean, len(samples)) and margin > 4.0:
                    continue
                cand.append((margin, qid))
            if not cand:
                break
            cand.sort()
            # take a small slice of the least-confident questions per round
            for _, qid in cand[:max(1, len(cand) // 8)]:
                if env.left() <= 0:
                    break
                samples = state[qid]
                v = len(samples) % nv
                a, s = env.sample(qid, v)
                samples.append((v, a, s))

    for qid, samples in state.items():
        out[qid] = _best_and_margin(samples)[0] if samples else ""

    for q in qs:
        out.setdefault(q["id"], "")
    env.submit(out)
