# Policy API

Your file `/app/policy.py` must define `run(env)`. It is imported and called once per batch inside a sandbox
(no network, no filesystem access beyond your own file, standard library only - **no numpy**). `print()` goes
to stderr and is shown to you by `lab evaluate` when something goes wrong.

    env.questions     # [{"id": "T0-017", "topic": "algebra"}, ...] - the whole batch, in a fixed order
    env.budget        # total samples available for the whole batch (shared pool)
    env.n_variants    # 5
    env.topics        # ["algebra", "geometry", "combinatorics", "number_theory"]

    answer, score = env.sample(qid, variant=0)
        # one fresh generation for that question through that prompt template, plus its verifier score.
        # answer is a string; score is a float. Costs 1 from the pool. Raises RuntimeError when the pool is
        # empty, so keep track: env.left() returns what is left.

    env.submit({qid: answer_string, ...})
        # call exactly once, at the end. Questions you omit count as wrong.

Minimal example (this is the current production policy):

    def run(env):
        out = {}
        per = env.budget // len(env.questions)
        for q in env.questions:
            counts = {}
            for _ in range(per):
                a, s = env.sample(q["id"], 0)
                counts[a] = counts.get(a, 0) + 1
            out[q["id"]] = max(counts, key=counts.get)
        env.submit(out)

Notes on the service (from the serving team):

* the verifier is a separately trained reward model. It was trained on a mixture of human preference data and
  solution traces, and nobody has audited it per topic;
* the 5 prompt templates were written by different people and are considered interchangeable; the service has
  always used template 0;
* answers are compared as exact strings after normalisation, so two samples agree iff their strings match.
