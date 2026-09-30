import heapq
import math

EASY_TOPICS = {"combinatorics", "number_theory"}
TEMP = 0.7


def _majority_counts(sample_list):
    counts = {}
    for a, _ in sample_list:
        counts[a] = counts.get(a, 0) + 1
    return counts


def _score_weighted_sums(sample_list):
    max_score = max(s for _, s in sample_list)
    sums = {}
    for a, s in sample_list:
        sums[a] = sums.get(a, 0.0) + math.exp((s - max_score) / TEMP)
    return sums


def _margin(agg):
    if not agg:
        return 0.0
    vals = sorted(agg.values(), reverse=True)
    total = sum(vals)
    if len(vals) == 1:
        return 1.0
    return (vals[0] - vals[1]) / total if total > 0 else 0.0


def _confidence(samples_for_q, is_easy):
    if not samples_for_q:
        return 0.0
    agg = _score_weighted_sums(samples_for_q) if is_easy else _majority_counts(samples_for_q)
    return _margin(agg)


def run(env):
    questions = env.questions
    n = len(questions)
    n_variants = env.n_variants
    budget = env.budget

    is_easy = {q["id"]: (q["topic"] in EASY_TOPICS) for q in questions}
    qids = [q["id"] for q in questions]

    samples = {qid: [] for qid in qids}
    next_variant = {qid: 0 for qid in qids}

    def take(qid):
        if env.left() <= 0:
            return False
        v = next_variant[qid] % n_variants
        a, s = env.sample(qid, v)
        samples[qid].append((a, s))
        next_variant[qid] += 1
        return True

    # Guarantee every question gets at least one sample up front, so we
    # always have something to submit even under a very tight budget.
    for qid in qids:
        if not take(qid):
            break

    avg = budget / n if n else 0
    phase1_extra = max(0, min(n_variants, int(round(avg / 2))) - 1)
    cap = max(phase1_extra + 1, min(25, int(round(avg * 3))))

    # Phase 1: top up every question to a small fixed baseline.
    for qid in qids:
        for _ in range(phase1_extra):
            if not take(qid):
                break

    # Phase 2: adaptively spend the shared pooled budget on the least
    # confident questions. Confidence is measured differently per topic
    # family: for combinatorics/number_theory the verifier score is
    # reliable, so we use the score-weighted margin; for algebra/geometry
    # the verifier is actively misleading (it agrees with the model's
    # systematic wrong answer more than with the correct one), so we use
    # the raw vote-count margin instead.
    heap = []
    version = {qid: 0 for qid in qids}
    for qid in qids:
        if len(samples[qid]) < cap:
            m = _confidence(samples[qid], is_easy[qid])
            heapq.heappush(heap, (m, qid, version[qid]))

    while env.left() > 0 and heap:
        m, qid, ver = heapq.heappop(heap)
        if ver != version[qid] or len(samples[qid]) >= cap:
            continue
        if not take(qid):
            break
        version[qid] += 1
        if len(samples[qid]) < cap:
            new_m = _confidence(samples[qid], is_easy[qid])
            heapq.heappush(heap, (new_m, qid, version[qid]))

    # Leftover budget (e.g. everything capped): spread round robin.
    idx = 0
    while env.left() > 0 and qids:
        qid = qids[idx % len(qids)]
        if not take(qid):
            break
        idx += 1

    out = {}
    for qid in qids:
        s = samples[qid]
        if not s:
            continue
        if is_easy[qid]:
            agg = _score_weighted_sums(s)
        else:
            agg = _majority_counts(s)
        out[qid] = max(agg.items(), key=lambda kv: kv[1])[0]

    env.submit(out)
