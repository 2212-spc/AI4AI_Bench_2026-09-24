from collections import defaultdict

def _choose(samples, topic):
    by = defaultdict(list)
    for a, s, v in samples:
        by[a].append(s)
    vals = {}
    for a, ss in by.items():
        if topic in ('combinatorics', 'number_theory'):
            vals[a] = sum(ss)
        else:
            vals[a] = sum(1 for s in ss if s > 0.3)
    best = max(vals.values())
    for a, _, _ in samples:
        if vals[a] == best:
            return a
    return samples[0][0]

def run(env):
    qs = env.questions
    out = {}
    plans = {
        'algebra': [2, 1, 1, 0, 2],
        'geometry': [3, 2, 0, 0, 1],
        'combinatorics': [1, 3, 0, 2, 0],
        'number_theory': [0, 0, 0, 3, 3],
    }
    # These plans use six calls per question and therefore exactly fill the pool.
    for q in qs:
        topic = q.get('topic')
        plan = plans.get(topic, [2, 1, 1, 1, 1])
        got = []
        for v, n in enumerate(plan):
            for _ in range(n):
                a, s = env.sample(q['id'], variant=v)
                got.append((a, s, v))
        out[q['id']] = _choose(got, topic)
    env.submit(out)
