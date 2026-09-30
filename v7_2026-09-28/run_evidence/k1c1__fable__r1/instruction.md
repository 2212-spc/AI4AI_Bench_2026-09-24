# Estimate a model's true win rate when your judge is biased

`cand-4` has been evaluated against the current production model on an arena of **900 prompts**. Every
prompt was scored by an **LLM judge**, and the judge says `cand-4` wins **42.9%** of the time. Nobody
believes that number: this judge is known to be wrong at different rates on different kinds of prompt, and
wrong in different directions depending on who actually won.

Each prompt carries two attributes you can see: **topic** (`code, math, writing, factual`) and **length** (`short, long`).

Your job is to report the win rate **on the production traffic mix**, which is not the arena's mix - the
arena over-samples some strata and under-samples others, and the production mix is given to you exactly.

## The lab

Run `/app/bin/lab spec` first. `lab view` is free and returns every prompt in a dev arena with its topic,
length and the judge's verdict. `lab label` buys **human ground truth** for prompt ids you choose - this is
the only unbiased signal in the task, and your entire budget is **220 annotations** for an arena of 900
prompts. `lab dry_run` runs your current `/app/estimator.py` on a dev arena and shows you what it produced,
for 12 annotations. It does **not** tell you whether the interval covered; that is what the labels are for.

## Deliverable: `/app/estimator.py`

A module defining `estimate(env)` (see `/app/docs/estimator_api.md`). It is run in a sandbox on **24 hidden
arenas** drawn from the same process, with the **stratum mix re-drawn for each one**. On each hidden arena
your estimator gets the full judge verdicts for free and may buy up to 220 human labels **of its own
choosing** - so an answer that works only because you hand-picked prompts in the lab will not transfer. It
must call `env.submit(point, lo, hi)` with a point estimate and a **90% interval** for the true
production-mix win rate.

## How it is graded (mechanically, against the simulator; only the file counts)

1. **R0** - `estimator.py` runs to completion and submits on all 24 hidden arenas.
2. **R1** - the interval covers the true production-mix win rate on at least **80%** of them.
3. **R2** - the mean interval width is at most **0.04**.

R2 exists to stop the obvious dodge: an interval of `[0, 1]` covers everything and says nothing. A width
under the cap is not itself worth points - covering is. Aim for an interval that is as wide as your
uncertainty actually is, and no wider.

Only `/app/estimator.py` is graded. Write it, check it with `lab dry_run`, then stop.
