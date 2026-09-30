"""Test-time sampling policy for solver-7b."""

from collections import defaultdict


def run(env):
    budget_total = env.budget
    questions = env.questions
    n_q = len(questions)

    # Samples collected per question: list of (variant, answer_str, score_float)
    samples = defaultdict(list)

    # Priority order for initial variants by topic
    def get_next_variant(qid, topic):
        count = len(samples[qid])
        if topic == "algebra":
            base_order = [1, 2, 0, 3, 4]
        elif topic == "number_theory":
            base_order = [2, 1, 0, 4, 3]
        else:
            base_order = [0, 1, 2, 3, 4]
        return base_order[count % 5]

    def sample_one(q):
        qid = q["id"]
        v = get_next_variant(qid, q.get("topic", ""))
        a, s = env.sample(qid, v)
        samples[qid].append((v, a, s))

    def is_satisfied(q):
        qid = q["id"]
        t = q.get("topic", "")
        q_samples = samples[qid]
        if not q_samples:
            return False

        if t in ("algebra", "number_theory"):
            # The verifier is well-calibrated: any correct answer has score > 0.4
            # and wrong answers rarely exceed 0.3.
            return any(s >= 0.5 for _, _, s in q_samples) or len(q_samples) >= 10
        else:
            # geometry and combinatorics:
            # Common wrong answers ("attractors") are overscored (> 1.45).
            # Correct answers typically score in [0.2, 1.35].
            # Satisfied if we have heard multiple variants (len >= 5) and
            # at least 2 samples for the same answer fall into [0.1, 1.35],
            # or we hit a cap of 12 samples.
            good_counts = defaultdict(int)
            for _, a, s in q_samples:
                if 0.1 <= s <= 1.35:
                    good_counts[a] += 1
            has_two_good = any(c >= 2 for c in good_counts.values())
            return (len(q_samples) >= 5 and has_two_good) or len(q_samples) >= 12

    # Pass 1: Give 2 initial samples to every question
    for q in questions:
        if env.left() <= 0:
            break
        sample_one(q)
    for q in questions:
        if env.left() <= 0:
            break
        sample_one(q)

    # Pass 2: Iteratively satisfy questions
    while env.left() > 0:
        needed = [q for q in questions if not is_satisfied(q)]
        if not needed:
            # If all questions meet basic satisfaction, distribute remaining
            # budget preferentially to geometry and combinatorics questions,
            # which benefit most from higher consensus.
            needed = sorted(
                [q for q in questions if q.get("topic") in ("geometry", "combinatorics")],
                key=lambda q: len(samples[q["id"]]),
            )
            if not needed or len(samples[needed[0]["id"]]) >= 15:
                needed = sorted(questions, key=lambda q: len(samples[q["id"]]))

        allocated_any = False
        for q in needed:
            if env.left() <= 0:
                break
            sample_one(q)
            allocated_any = True

        if not allocated_any:
            break

    # Final prediction selection
    out = {}
    for q in questions:
        qid = q["id"]
        t = q.get("topic", "")
        q_samples = samples[qid]

        if not q_samples:
            # Fallback if somehow no samples were drawn
            a, s = env.sample(qid, 0)
            q_samples.append((0, a, s))

        if t in ("algebra", "number_theory"):
            # Verifier is highly reliable for algebra and number theory
            best = max(q_samples, key=lambda x: x[2])
            out[qid] = best[1]
        else:
            # geometry and combinatorics:
            # Penalize overscored attractors (s > 1.45)
            # Reward realistic correct scores around 1.0 (in [0.1, 1.35])
            weights = defaultdict(float)
            for _, a, s in q_samples:
                if s > 1.45:
                    w = -2.0
                elif s > 1.30:
                    w = 0.1
                elif s >= 0.1:
                    w = 2.0 - abs(s - 1.0)
                else:
                    w = 0.0
                weights[a] += w

            best_ans = max(weights, key=weights.get)
            # If all candidates have non-positive weights, fallback to answer closest to 1.0
            if weights[best_ans] <= 0:
                best_ans = min(q_samples, key=lambda x: abs(x[2] - 1.0))[1]
            out[qid] = best_ans

    env.submit(out)
