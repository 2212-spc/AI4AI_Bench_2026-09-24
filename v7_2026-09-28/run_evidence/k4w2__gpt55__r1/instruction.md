# Test-time sampling policy for a math-answering service

`solver-7b` answers competition-style math questions. It is stochastic: you can sample it several times per
question, and a separate **verifier model** scores every candidate answer (higher = the verifier believes the
answer more). Each question can be asked through one of **5 interchangeable prompt templates** (`variant`
0-4); the templates are paraphrases of the same instruction, and the service treats them as equivalent.

Serving is capacity-limited: for a batch of 400 questions you get a pool of **2400 samples in total** (6 per
question on average). The pool is shared - you may spend 2 samples on one question and 12 on another - and the
service currently just takes a plain majority vote over 6 samples of template 0. You have been asked to
replace that with a better policy.

## Your job

Write **`/app/policy.py`**, a Python file exposing `run(env)` (see `/app/docs/policy_api.md`; a working
majority-vote implementation is already there). It receives the batch, may call `env.sample(qid, variant)` up
to the pooled budget, and must call `env.submit({question_id: answer_string})` exactly once. It runs in a
sandbox with no network and no access to this directory - only the standard library, and `numpy` is *not*
available.

## The lab

Run `/app/bin/lab spec` first. You have 10000 credits.

* `lab dev_sample n=20 topic=algebra variants='[0,1,2,3,4]' k=2` - draws fresh **labelled** dev questions from
  the same service and returns, for each, the ground-truth answer and the requested samples with their
  verifier scores. Costs `n * len(variants) * k` credits.
* `lab evaluate` - runs your current `/app/policy.py` on a fresh batch of 400 dev questions under the real
  budget and returns its accuracy overall and per topic (plus stderr if it crashes). Costs 1000 credits, so you
  can afford a handful of these - use dev samples to understand the service, not `evaluate` as a search loop.

Example:

    /app/bin/lab dev_sample n=15 topic=geometry variants='[0,1,2]' k=2

## How it is graded (mechanically)

Your `policy.py` is run on **3 fresh hidden batches** of 400 questions from the same service, with the same
pooled budget, and scored by

    s = (your accuracy - majority@6 accuracy) / (reference policy accuracy - majority@6 accuracy)

where both reference numbers are measured on those same hidden batches. You pass if

1. **R0** - the policy runs to completion and submits an answer for every question on all 3 batches, without
   crashing and without exceeding the sample pool, and
2. **R1** - `s >= 0.85`.

On the dev distribution the plain majority@6 baseline scores around 0.51 and the reference policy around
0.92 accuracy, so R1 needs roughly **0.86** accuracy or better. Both bars are re-measured on the hidden
batches, so aim to beat the reference, not to match a number.

Only `/app/policy.py` is graded. Write it, make sure `lab evaluate` runs it without errors, then stop.
