# Labelling services

All verdicts are binary: 1 = the response is correct, 0 = it is not.

* **LLM judge** (already run on every response; `judge_A`, `judge_B` in `pool.csv`).  A strong general model
  prompted with the question and a grading rubric.
* **crowd** - 0.05 credits per vote.  A vetted crowd worker reads the question and the response and votes.  Every
  vote comes from a new worker, so asking again for the same response gives an independent vote.
* **review** - 0.10 credits per response.  A domain expert reads the response and records a verdict (the review
  queue budgets about two minutes per response).  The team keeps one recorded verdict per response: reviewing
  the same response again returns the same verdict (and is charged again).
* **verify** - 2.00 credits per response.  Full verification: an expert independently re-derives the answer,
  runs the code, and checks every claim against sources.  This is the review board's definition of "correct".
  Verification is deterministic: verifying the same response again returns the same verdict (and is charged
  again).

Response lengths in `pool.csv` are in tokens.
