# Best approach: score-based for reliable topics, hybrid for others
def run(env):
    out = {}
    n_questions = len(env.questions)
    samples_per_q = env.budget // n_questions

    # Topics where verifier score is reliable
    score_based_topics = {'combinatorics', 'number_theory'}

    for q in env.questions:
        qid = q["id"]
        topic = q["topic"]
        answer_info = {}  # answer -> {'count': int, 'scores': [float]}

        # Sample across all variants
        for i in range(samples_per_q):
            variant = i % env.n_variants
            answer, score = env.sample(qid, variant)

            if answer not in answer_info:
                answer_info[answer] = {'count': 0, 'scores': []}

            answer_info[answer]['count'] += 1
            answer_info[answer]['scores'].append(score)

        # Choose strategy based on topic
        if topic in score_based_topics:
            # For reliable topics: weight by total score
            best_answer = max(answer_info, key=lambda a: sum(answer_info[a]['scores']))
        else:
            # For other topics: use frequency + score as tiebreaker
            def score_fn(a):
                count = answer_info[a]['count']
                avg_score = sum(answer_info[a]['scores']) / len(answer_info[a]['scores'])
                # Tuple comparison: count primary, avg_score secondary
                return (count, avg_score)
            best_answer = max(answer_info, key=score_fn)

        out[qid] = best_answer

    env.submit(out)

