"""Topic-calibrated answer selection with a shared adaptive sampling pool."""
import heapq
import math


def _logadd(a, b):
    m = max(a, b)
    return m + math.log1p(math.exp(min(a, b) - m))


class _Question:
    def __init__(self, q, index):
        self.qid = q['id']
        self.topic = q['topic']
        self.tricky = self.topic in ('geometry', 'combinatorics')
        self.groups = {}
        self.n = 0
        self.counts = [0] * 5
        self.answer = ''
        self.confidence = 0.0
        self.index = index

    def add(self, answer, score, variant):
        # Each answer has one latent class. Accumulate evidence over all of its
        # scores, so a single unusually flattering score cannot dominate.
        g = self.groups.setdefault(answer, [0, 0.0, 0.0, 0.0, set()])
        g[0] += 1
        g[1] += -0.5 * ((score - 1.0) / .35) ** 2
        noise_mean = -.55 if self.topic == 'algebra' else -.65 if self.topic == 'number_theory' else -.60
        g[2] += -0.5 * ((score - noise_mean) / .35) ** 2
        g[3] += -0.5 * ((score - 1.82) / .35) ** 2
        g[4].add(variant)
        self.n += 1
        self.counts[variant] += 1
        weights = []
        for a, h in self.groups.items():
            if self.tricky:
                wrong = _logadd(math.log(1.2) + h[2], math.log(.8) + h[3])
                # A wrong answer may be tied to a prompt. Agreement between
                # prompts is additional, modest evidence for correctness.
                logodds = h[1] - wrong + (math.log(1.8) if len(h[4]) > 1 else 0.0)
            else:
                logodds = h[1] - h[2] - math.log(2.0)
            weights.append((logodds, a))
        best, self.answer = max(weights)
        scale = max(0.0, best)
        denom = math.exp(-scale) + sum(math.exp(w - scale) for w, _ in weights)
        self.confidence = math.exp(best - scale) / denom

    def priority(self):
        # Failed searches increasingly indicate a difficult question. Discount
        # their expected return, while continuing to reserve them some effort.
        recovery = .4 / (1.0 + self.n / 4.0) + .3 * self.confidence
        return (1.0 - self.confidence) * recovery

    def variant(self, n_variants):
        # Balance templates; change the first template across questions too.
        return min(range(n_variants), key=lambda v: (self.counts[v], (v - self.index) % n_variants))


def run(env):
    states = [_Question(q, i) for i, q in enumerate(env.questions)]
    heap = []
    for i, state in enumerate(states):
        if env.left() > 0:
            v = state.variant(env.n_variants)
            answer, score = env.sample(state.qid, v)
            state.add(answer, score, v)
        heapq.heappush(heap, (-state.priority(), i))
    while heap and env.left() > 0:
        _, i = heapq.heappop(heap)
        state = states[i]
        v = state.variant(env.n_variants)
        answer, score = env.sample(state.qid, v)
        state.add(answer, score, v)
        heapq.heappush(heap, (-state.priority(), i))
    env.submit({s.qid: s.answer for s in states})
