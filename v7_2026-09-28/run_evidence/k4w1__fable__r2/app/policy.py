# Test-time sampling policy for solver-7b.
#
# Findings from labelled dev data:
#  * combinatorics / number_theory: the verifier is essentially perfect - any sample with score > ~0.5 is
#    correct, wrong samples score < 0.2.  Sampling until a high-score sample appears is cheap (~3 samples).
#  * geometry / algebra: the verifier is *non-monotonic*.  Correct answers score ~N(1.0, 0.4); wrong answers
#    are either low-score noise (< 0.2, unique strings) or confident "trap" answers scoring ~1.5-2.5.  Traps
#    are mostly tied to a single prompt template, whereas the correct answer recurs across templates.  So we
#    (a) weight samples by a per-topic likelihood ratio derived from the score, and (b) reward answers that
#    show up under several distinct templates while discounting repeats within the same template.
#  * Difficulty varies a lot per question, so the shared pool is allocated adaptively: a cheap first pass,
#    then the remaining budget goes to the least-confident questions.
import math

TH_NOISE = 0.2          # below this a sample is noise in every topic

# log P(score | correct) / P(score | wrong) per score bin, per topic family (from dev histograms, smoothed)
_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
_LLR = {
    "geometry":      [-4.2, -0.3, 1.5, 3.0, 3.5, 3.2, 1.8, 0.8, 0.0, -0.8, -3.0, -3.6],
    "algebra":       [-4.2, 0.0, 0.9, 1.7, 2.8, 1.9, 1.8, 1.2, 0.1, -0.5, -1.5, -3.6],
    "combinatorics": [-4.2, -0.6, 0.5, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
    "number_theory": [-4.2, -0.6, 0.5, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
}
_DEFAULT_LLR = _LLR["algebra"]

# extra evidence for each additional distinct template an answer is seen under (non-noise samples)
VARIANT_BONUS = 1.5
# repeats of an answer inside the same template are worth this fraction of a fresh observation
REPEAT_WEIGHT = 0.35

EASY_TOPICS = ("combinatorics", "number_theory")
EARLY_CONF = 6.0     # phase-1 early stop confidence (hard topics)
DONE_CONF = 7.0      # phase-2 stop confidence (hard topics)
MAX_PER_Q = 20
PHASE1_HARD = 3


def llr(topic, score):
    tab = _LLR.get(topic, _DEFAULT_LLR)
    i = 0
    while i < len(_BINS) and score >= _BINS[i]:
        i += 1
    return tab[i]


class QState:
    __slots__ = ("qid", "topic", "samples", "n_by_variant", "easy")

    def __init__(self, qid, topic):
        self.qid = qid
        self.topic = topic
        self.samples = []             # (variant, answer, score)
        self.n_by_variant = [0] * 5
        self.easy = topic in EASY_TOPICS

    # ---- evidence ----
    def candidates(self):
        """answer -> evidence (log-odds-ish)."""
        per = {}
        for v, a, s in self.samples:
            per.setdefault(a, {}).setdefault(v, []).append(s)
        out = {}
        for a, byv in per.items():
            ev = 0.0
            nv_good = 0
            for v, scores in byv.items():
                scores = sorted(scores, reverse=True)
                w = 1.0
                for s in scores:
                    ev += w * llr(self.topic, s)
                    w = REPEAT_WEIGHT
                if max(scores) > TH_NOISE:
                    nv_good += 1
            if nv_good > 1:
                ev += VARIANT_BONUS * (nv_good - 1)
            out[a] = ev
        return out

    def best(self):
        c = self.candidates()
        if not c:
            return None, -99.0, -99.0
        ranked = sorted(c.items(), key=lambda kv: -kv[1])
        top_a, top_e = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -6.0
        return top_a, top_e, second

    def confidence(self):
        """higher = more sure.  Margin over runner-up, capped by absolute evidence."""
        a, e, s2 = self.best()
        if a is None:
            return -99.0
        return min(e - s2, e + 3.0)

    def next_variant(self):
        # least-sampled template first; ties by index (template 0 is slightly best on geometry)
        m = min(self.n_by_variant)
        for v in range(5):
            if self.n_by_variant[v] == m:
                return v
        return 0

    def add(self, v, a, s):
        self.samples.append((v, a, s))
        self.n_by_variant[v] += 1

    def n(self):
        return len(self.samples)


def draw(env, st):
    v = st.next_variant()
    a, s = env.sample(st.qid, v)
    st.add(v, a, s)


def run(env):
    qs = [QState(q["id"], q.get("topic", "")) for q in env.questions]
    nq = len(qs)
    budget = env.budget
    used = 0

    def left():
        return budget - used

    # ---- phase 1: cheap first pass ----
    # easy topics: sample until one clearly-correct sample (score > 0.5), at most 3 for now
    # hard topics: one sample from each of the first 3 templates, stopping early when confident
    for st in qs:
        if st.easy:
            cap = 3
            while st.n() < cap and left() > 0:
                draw(env, st); used += 1
                if max(s for _, _, s in st.samples) > 0.5:
                    break
        else:
            cap = PHASE1_HARD
            while st.n() < cap and left() > 0:
                draw(env, st); used += 1
                if st.n() >= 2 and st.confidence() >= EARLY_CONF:
                    break

    # ---- phase 2: spend the rest on the least-confident questions ----
    # a question is "done" when confident enough; thresholds differ by family
    def done(st):
        if st.n() >= MAX_PER_Q:
            return True
        if st.easy:
            return max(s for _, _, s in st.samples) > 0.5
        return st.confidence() >= DONE_CONF

    import heapq
    heap = []
    for i, st in enumerate(qs):
        if not done(st):
            heapq.heappush(heap, (st.confidence(), i))
    while left() > 0 and heap:
        conf, i = heapq.heappop(heap)
        st = qs[i]
        draw(env, st); used += 1
        if not done(st):
            heapq.heappush(heap, (st.confidence(), i))

    # ---- submit ----
    out = {}
    for st in qs:
        a, _, _ = st.best()
        if a is None:
            a = "0"
        out[st.qid] = a
    env.submit(out)
