# Estimator API

Your file `/app/estimator.py` must define `estimate(env)`. It is imported and called once per hidden arena
inside a sandbox (no network, no filesystem access beyond your own file, standard library only - **no
numpy**). `print()` goes to stderr and is shown to you by `lab dry_run` when something goes wrong.

    env.rows              # list of dicts, one per prompt in THIS arena:
                          #   {"id": int, "topic": str, "length": str, "judge_win": bool}
                          # the judge's verdict is free - it already ran on everything
    env.label_budget      # how many human labels you may buy on this arena
    env.topics            # the topic values
    env.lengths           # the length values
    env.target_coverage   # the nominal coverage of the interval you are asked for
    env.production_mix    # {"topic|length": weight} - the mix you are estimating ON, sums to 1

    got = env.label([id1, id2, ...])   # -> {"<id>": True/False}; True = the candidate really won
                                       # raises if you exceed env.label_budget

    env.submit(point, lo, hi)          # report and return; call exactly once

Notes:

* `env.production_mix` is **re-drawn for every arena** and is different from the arena's own stratum shares.
  Read it at run time; do not hardcode what you saw in the lab.
* The judge's error rates are properties of the judge, not of the arena - they are the same on every arena.
  Which prompts you have to spend labels on to *measure* them is up to you.
* Nothing in `env` reveals the truth for an unlabelled prompt, and the sandbox has no filesystem access;
  attempts to read it out of the environment are recorded and graded as a failure.
