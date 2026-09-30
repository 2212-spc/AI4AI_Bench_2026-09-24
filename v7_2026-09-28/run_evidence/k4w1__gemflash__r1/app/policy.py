"""Policy for test-time sampling and verification.

Key observations:
1. Combinatorics & Number Theory:
   The verifier scores are reliable. Wrong answers rarely score > 0.5.
   Any score >= 0.70 is virtually guaranteed to be correct (>99% accuracy).
   We can sequentially sample cycling prompt variants 0..4 and stop as soon
   as a sample scores >= 0.70. If no sample reaches 0.70, we select the answer
   with the highest verifier score.

2. Algebra & Geometry:
   The verifier suffers from overconfidence on wrong answers (hallucinations scoring > 1.6).
   However, true correct answers have scores tightly centered around ~1.0 (sweet spot [0.35, 1.35]).
   Scores > 1.6 are overwhelmingly wrong hallucinations.
   Furthermore, prompting with diverse paraphrased variants (0..4) eliminates variant-specific errors.
   We evaluate answers using:
   - sweet-spot weight (+1.0 for scores in [0.35, 1.35])
   - penalty (-1.0 for scores > 1.60)
   - cross-variant consistency (number of distinct prompt templates producing the answer)
"""


def _score_alg_geom(samples):
    ans_val = {}
    ans_vars = {}
    ans_counts = {}
    for s in samples:
        a = s["answer"]
        sc = s["score"]
        v = s["variant"]
        ans_counts[a] = ans_counts.get(a, 0) + 1
        if 0.35 <= sc <= 1.35:
            w = 1.0
        elif sc > 1.60:
            w = -1.0
        else:
            w = 0.0
        ans_val[a] = ans_val.get(a, 0.0) + w
        if a not in ans_vars:
            ans_vars[a] = set()
        if 0.35 <= sc <= 1.45:
            ans_vars[a].add(v)
    return max(
        ans_val.keys(),
        key=lambda a: (ans_val[a], len(ans_vars.get(a, set())), ans_counts[a]),
    )


def _check_alg_geom_early(samples):
    ans_vars = {}
    ans_high = {}
    for s in samples:
        a = s["answer"]
        sc = s["score"]
        v = s["variant"]
        if 0.35 <= sc <= 1.35:
            ans_vars.setdefault(a, set()).add(v)
        elif sc > 1.60:
            ans_high[a] = ans_high.get(a, 0) + 1
    for a, vs in ans_vars.items():
        if len(vs) >= 3 and ans_high.get(a, 0) == 0:
            return a
    return None


def run(env):
    q_state = {}
    n_vars = getattr(env, "n_variants", 5)

    for q in env.questions:
        q_state[q["id"]] = {
            "id": q["id"],
            "topic": q["topic"],
            "samples": [],
            "resolved": False,
            "chosen": None,
        }

    def sample_one(qid):
        if env.left() <= 0:
            return None
        st = q_state[qid]
        var = len(st["samples"]) % n_vars
        ans, sc = env.sample(qid, var)
        s_obj = {"variant": var, "answer": ans, "score": sc}
        st["samples"].append(s_obj)
        return s_obj

    # Phase 1: Combinatorics & Number Theory
    # Sequential sampling up to 8 samples, stopping immediately on score >= 0.70
    for q in env.questions:
        qid = q["id"]
        st = q_state[qid]
        if st["topic"] in ("combinatorics", "number_theory"):
            for _ in range(8):
                if env.left() <= 0:
                    break
                s = sample_one(qid)
                if s and s["score"] >= 0.70:
                    st["resolved"] = True
                    st["chosen"] = s["answer"]
                    break

    # Phase 2: Algebra & Geometry initial 5 samples (one per variant 0..4)
    for q in env.questions:
        qid = q["id"]
        st = q_state[qid]
        if st["topic"] in ("algebra", "geometry"):
            for _ in range(min(5, n_vars)):
                if env.left() <= 0:
                    break
                sample_one(qid)
            cand = _check_alg_geom_early(st["samples"])
            if cand is not None:
                st["resolved"] = True
                st["chosen"] = cand

    # Phase 3: Round robin additional samples for unresolved Algebra & Geometry
    while env.left() > 0:
        added = False
        for q in env.questions:
            if env.left() <= 0:
                break
            qid = q["id"]
            st = q_state[qid]
            if st["topic"] in ("algebra", "geometry") and not st["resolved"]:
                s = sample_one(qid)
                if s is not None:
                    added = True
                    cand = _check_alg_geom_early(st["samples"])
                    if cand is not None:
                        st["resolved"] = True
                        st["chosen"] = cand
        if not added:
            break

    # Phase 4: Any remaining budget to unresolved Combinatorics & Number Theory
    while env.left() > 0:
        added = False
        for q in env.questions:
            if env.left() <= 0:
                break
            qid = q["id"]
            st = q_state[qid]
            if st["topic"] in ("combinatorics", "number_theory") and not st["resolved"]:
                s = sample_one(qid)
                if s is not None:
                    added = True
                    if s["score"] >= 0.70:
                        st["resolved"] = True
                        st["chosen"] = s["answer"]
        if not added:
            break

    # Phase 5: Any leftover budget distributed to any unresolved question
    while env.left() > 0:
        added = False
        for q in env.questions:
            if env.left() <= 0:
                break
            qid = q["id"]
            st = q_state[qid]
            if not st["resolved"]:
                s = sample_one(qid)
                if s is not None:
                    added = True
        if not added:
            break

    # Final decision for all questions
    out = {}
    for q in env.questions:
        qid = q["id"]
        st = q_state[qid]
        if not st["resolved"]:
            if not st["samples"]:
                # Fallback if no samples were drawn
                var = 0
                ans, sc = env.sample(qid, var)
                st["samples"].append({"variant": var, "answer": ans, "score": sc})
            if st["topic"] in ("combinatorics", "number_theory"):
                st["chosen"] = max(st["samples"], key=lambda s: s["score"])["answer"]
            else:
                st["chosen"] = _score_alg_geom(st["samples"])
            st["resolved"] = True
        out[qid] = st["chosen"]

    env.submit(out)
