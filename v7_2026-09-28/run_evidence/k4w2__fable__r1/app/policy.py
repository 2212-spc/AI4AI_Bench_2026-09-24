"""Adaptive test-time sampling policy for solver-7b.

Findings on the dev service that drive this policy:
  * algebra / number_theory: the verifier is essentially perfect.  Any sample scoring above ~0.3 is
    correct, so we sample until we see one and stop.
  * geometry / combinatorics: the verifier is miscalibrated.  Scores in roughly [0.3, 1.3] are almost
    always correct, scores above ~1.6 are usually a confidently-wrong "seductive" answer, scores below
    0.3 are wrong.  Wrong answers also cluster within a prompt template but rarely across templates,
    so we rotate templates and reward cross-template agreement.
  * Budget freed on the easy topics is spent on the hard ones, question by question, wherever the
    current answer is least certain.
"""
import math
from collections import defaultdict

EASY = {"algebra", "number_theory"}

# P(correct | score) lookup for the hard topics (bin upper edge, probability)
HARD_TABLE = [
    (0.0, 0.01), (0.3, 0.25), (0.5, 0.90), (0.7, 0.96), (0.9, 0.97), (1.1, 0.95),
    (1.3, 0.84), (1.5, 0.46), (1.7, 0.26), (1.9, 0.08), (2.2, 0.03), (99.0, 0.02),
]
EASY_TABLE = [(0.0, 0.005), (0.3, 0.2), (0.5, 0.95), (99.0, 0.995)]

EASY_STOP = 0.35      # easy topic: stop as soon as a sample scores at least this
EASY_MAX = 8          # ... or after this many samples
HARD_MIN = 2          # hard topic: minimum samples before it may be considered settled
HARD_CONF = 0.985     # hard topic: stop when posterior of leader reaches this
VARIANT_BONUS = 1.2   # extra log-odds per additional distinct template agreeing


def _p(table, s):
    for hi, p in table:
        if s < hi:
            return p
    return table[-1][1]


def _llr(table, s):
    p = min(max(_p(table, s), 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def _posterior(samples, table):
    """samples: list of (answer, score, variant). Returns (best_answer, confidence)."""
    if not samples:
        return None, 0.0
    logit = defaultdict(float)
    variants = defaultdict(set)
    for a, s, v in samples:
        logit[a] += _llr(table, s)
        variants[a].add(v)
    for a in logit:
        logit[a] += VARIANT_BONUS * (len(variants[a]) - 1)
    # prior: unseen "other" answer mass
    m = max(logit.values())
    z = sum(math.exp(l - m) for l in logit.values()) + math.exp(-m)
    best = max(logit, key=logit.get)
    return best, math.exp(logit[best] - m) / z


def run(env):
    qs = env.questions
    n = len(qs)
    state = {q["id"]: [] for q in qs}
    topic = {q["id"]: q["topic"] for q in qs}
    nxt = {q["id"]: (i % env.n_variants) for i, q in enumerate(qs)}
    hard_ids = [q["id"] for q in qs if topic[q["id"]] not in EASY]
    easy_ids = [q["id"] for q in qs if topic[q["id"]] in EASY]

    def draw(qid):
        v = nxt[qid]
        nxt[qid] = (v + 1) % env.n_variants
        try:
            a, s = env.sample(qid, v)
        except RuntimeError:
            return False
        state[qid].append((str(a), float(s), v))
        return True

    def easy_done(qid):
        st = state[qid]
        return any(s >= EASY_STOP for _, s, _ in st) or len(st) >= EASY_MAX

    # Phase 0: one sample for everyone (guarantees an answer for every question).
    for q in qs:
        if env.left() <= 0:
            break
        draw(q["id"])

    # Phase 1: easy topics - keep sampling until a trusted answer shows up.
    active = [qid for qid in easy_ids if not easy_done(qid)]
    while active and env.left() > 0:
        # reserve at least HARD_MIN-1 more samples per hard question
        reserve = sum(max(0, HARD_MIN - len(state[h])) for h in hard_ids)
        if env.left() <= reserve:
            break
        nxt_active = []
        for qid in active:
            if env.left() <= reserve:
                nxt_active.append(qid)
                continue
            draw(qid)
            if not easy_done(qid):
                nxt_active.append(qid)
        active = nxt_active

    # Phase 2: hard topics - bring everyone to HARD_MIN, then spend the rest where least certain.
    for qid in hard_ids:
        while len(state[qid]) < HARD_MIN and env.left() > 0:
            draw(qid)

    def table_for(qid):
        return EASY_TABLE if topic[qid] in EASY else HARD_TABLE

    def confidence(qid):
        return _posterior(state[qid], table_for(qid))[1]

    CAP = 40
    conf = {q["id"]: confidence(q["id"]) for q in qs}
    ids = [q["id"] for q in qs]
    # Phase 3: global allocation - always sample the least certain question, until its posterior clears
    # the bar; once everything clears the bar keep pushing the least certain one anyway.
    while env.left() > 0:
        cands = [qid for qid in ids if conf[qid] < HARD_CONF and len(state[qid]) < CAP]
        if not cands:
            cands = [qid for qid in ids if len(state[qid]) < CAP]
            if not cands:
                break
        qid = min(cands, key=lambda q: conf[q])
        if not draw(qid):
            break
        conf[qid] = confidence(qid)

    out = {}
    for q in qs:
        qid = q["id"]
        table = EASY_TABLE if topic[qid] in EASY else HARD_TABLE
        best, _ = _posterior(state[qid], table)
        out[qid] = best if best is not None else ""
    env.submit(out)
