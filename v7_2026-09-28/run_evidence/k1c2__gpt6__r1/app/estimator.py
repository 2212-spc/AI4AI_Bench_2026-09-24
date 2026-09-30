"""Production-weighted, judge-stratified human audit.

Development calibration: 13 random labels in each topic/length/judge cell,
across two independent arenas. Only aggregate error counts are retained.
The conditional error probabilities can change with prevalence, so these
observations receive 3/4 weight. A weak six-observation prior pools the
observed overall error frequency without equating the two error directions.
Intervals describe the weighted finite arena, including unobserved outcomes.
"""
import random


_DEV_ERRORS = {
    ('code', 'long'): (0, 1),
    ('code', 'short'): (1, 0),
    ('factual', 'long'): (2, 1),
    ('factual', 'short'): (0, 0),
    ('math', 'long'): (0, 0),
    ('math', 'short'): (1, 0),
    ('writing', 'long'): (0, 1),
    ('writing', 'short'): (0, 1),
}


def estimate(env):
    rng = random.Random(731904)
    grouped = {}
    totals = {}
    for row in env.rows:
        s = (row['topic'], row['length'])
        key = s + (int(bool(row['judge_win'])),)
        grouped.setdefault(key, []).append(row['id'])
        totals[s] = totals.get(s, 0) + 1

    cells = []
    for key in sorted(grouped):
        topic, length, judge = key
        ids = sorted(grouped[key])
        rng.shuffle(ids)
        weight = float(env.production_mix.get(topic + '|' + length, 0.0))
        weight *= len(ids) / totals[(topic, length)]
        errors = _DEV_ERRORS.get((topic, length), (0, 0))[judge]
        # Beta prior on P(human disagrees | topic, length, judge).
        a = 0.25 + 0.75 * errors
        b = 5.75 + 0.75 * (13 - errors)
        cells.append(dict(ids=ids, n=len(ids), used=0, errors=0,
                          a=a, b=b, weight=weight, judge=judge))

    budget = min(int(env.label_budget), len(env.rows))
    # Three audit waves: learn where the judge fails, then devote remaining
    # labels to cells with the largest reduction in production uncertainty.
    remaining = budget
    while remaining:
        batch_size = min(remaining, max(1, (budget + 2) // 3))
        planned = [0] * len(cells)
        for _ in range(batch_size):
            best, best_gain = None, -1.0
            for i, c in enumerate(cells):
                m = c['n'] - c['used'] - planned[i]
                if m <= 0:
                    continue
                strength = c['a'] + c['b']
                p = c['a'] / strength
                after = strength + planned[i]
                gain = (c['weight'] / c['n']) ** 2 * p * (1-p)
                gain *= ((after + m) / (after + 1)) ** 2
                if gain > best_gain:
                    best, best_gain = i, gain
            if best is None:
                break
            planned[best] += 1
        ids = []
        for c, count in zip(cells, planned):
            ids.extend(c['ids'][c['used']:c['used'] + count])
        if not ids:
            break
        labels = {}
        for start in range(0, len(ids), 200):
            labels.update(env.label(ids[start:start+200]))
        for c, count in zip(cells, planned):
            selected = c['ids'][c['used']:c['used']+count]
            e = sum(bool(labels[str(i)] if str(i) in labels else labels[i])
                    != bool(c['judge']) for i in selected)
            c['a'] += e
            c['b'] += count - e
            c['errors'] += e
            c['used'] += count
        remaining -= len(ids)

    point = 0.0
    for c in cells:
        m = c['n'] - c['used']
        error_rate = (c['errors'] + m*c['a']/(c['a']+c['b'])) / c['n']
        point += c['weight'] * (1-error_rate if c['judge'] else error_rate)

    # Beta-binomial predictive draws retain asymmetry for sparse errors and
    # remove all uncertainty in fully audited cells (finite population).
    draws = []
    for _ in range(6000):
        value = 0.0
        for c in cells:
            m = c['n'] - c['used']
            errors = c['errors']
            if m:
                p = rng.betavariate(c['a'], c['b'])
                errors += sum(rng.random() < p for _ in range(m))
            rate = errors / c['n']
            value += c['weight'] * (1-rate if c['judge'] else rate)
        draws.append(value)
    draws.sort()
    tail = (1-float(env.target_coverage)) / 2
    lo = draws[max(0, min(len(draws)-1, int(tail*len(draws))))]
    hi = draws[max(0, min(len(draws)-1, int((1-tail)*len(draws))))]
    env.submit(point, min(point, lo), max(point, hi))
