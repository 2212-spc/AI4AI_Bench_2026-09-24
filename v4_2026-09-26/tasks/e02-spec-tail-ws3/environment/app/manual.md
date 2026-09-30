# ServeLab manual

## 1. What the service models (abstraction boundary)

ServeLab is a **simulated** inference-serving stack.  No real kernel is run; every request is answered by
a simulator of one fixed model on one fixed accelerator type.  The simulator is not a replica of any real
deployment: **its hardware and model constants were drawn fresh for this lab**, so published roofline
numbers, acceptance rates and quantization results describe other systems.  Measure; do not recall.

The simulator contains the following kinds of effects.  **Any given lab may switch some of them off
(held neutral); which ones are active is not stated.**

- *Arithmetic versus memory.*  A decode step costs the larger of its memory-traffic time and its
  arithmetic time, so throughput per accelerator rises with batch up to a point and then stops.
- *KV cache.*  Each sequence holds a cache proportional to its context length; weights, activations and
  caches share a fixed memory, which puts a hard ceiling on batch.  A request above the ceiling is refused.
- *Numeric precision.*  Serving at fewer bits changes both the memory traffic and the served quality.
- *Speculative decoding.*  A draft model proposes several tokens which the target verifies; how many are
  accepted can depend on the position within the proposal, and a decay measured over one proposal length
  does not have to continue at the same rate beyond it.  Drafting costs time per proposed token.
- *Queueing.*  Under a Poisson arrival process, mean end-to-end latency follows from utilisation and the
  variability of service times; latency quantiles are taken from an exponential sojourn-time distribution
  with that mean.  Utilisation is *not* something you set: it is the arrival rate times the mean service
  time, and the mean service time depends on the batch.
- *Test-time compute.*  Sampling k answers raises the chance that a correct answer is *among* them; what
  a verifier can actually *select* is a different and lower quantity.
- *Noise.*  Throughput and latency carry multiplicative run-to-run noise; accuracies and acceptance rates
  carry sampling noise over a fixed number of items.  A benchmark averages over the decode steps it had
  time for, so how long you run it is what buys precision.

Not modelled: kernel-level scheduling, network, tensor/pipeline parallel topology, tokenizer, disaggregated
prefill, cache reuse across requests.

## 2. Guarantees

- **Fixed laws.**  Every mechanism is a fixed law whose constants do not depend on which request you make.
  There are no hidden thresholds, **except** where Section 5 lists a *known unknown*.
- **Determinism.**  An identical request with the same seed returns identical numbers (and is charged
  again).  Different seeds give independent draws.  `batch_max`, `kv_bytes_per_seq` and `util` are exact.
- **Duration buys precision.**  `dur` is measured against a 20-accelerator-second reference run: the
  relative noise on `tokens_per_s`, `ms_per_token` and the latency quantiles scales as `1/sqrt(dur/20)`,
  and the number of proposals behind each reported acceptance position is proportional to `dur`.  A run
  costs what its `dur` says, so precision is something you buy.
- **Refusals are free.**  A configuration the service refuses (for example, out of memory) is not charged.
- **Notebook rows are real service responses.**  The team's notes are the team's interpretation and may be wrong.

## 3. How answers are graded

Every question is graded mechanically against this lab's hidden world.  Nothing you write outside the
answer file is read, and there is no credit for an explanation of a wrong answer.

- **Numeric questions** take an interval `{"lo": x, "hi": y}`.  The interval must be the *set of values
  consistent with everything that can be known*: what the lab can measure within your budget, plus the
  guarantees and documented ranges in this manual.
  - If the lab pins a quantity down, answer a degenerate interval (`lo == hi` = your estimate).  Seed
    noise is handled by the grading tolerance, so **do not** widen for statistical uncertainty.
  - If the quantity depends on a known unknown (Section 5), `lo`/`hi` are the minimum and maximum over
    the values of that unknown that are consistent with its documented range **and** with what the
    lab can measure.
  - Each endpoint is compared with the true endpoint.  The tolerance is calibrated to how precisely a
    well-designed set of experiments within the budget can pin that endpoint down.
- **Counterexample questions** ask whether a stated claim is *forced* by the evidence, and take
  `{"verdict": "refutable" | "entailed", "witness": {...}}`.
  - The question lists a few **free parameters**, each with a range.  A *witness* gives a value for every
    one of them and for nothing else: the witness world is this lab's own world with exactly those
    parameters moved, and everything else left as it is.
  - Answer `refutable` **with a witness** if some assignment both (a) reproduces the evidence and (b)
    makes the claim false.  Answer `entailed`, with no witness, if no assignment can do both - that is,
    if the claim follows from the evidence no matter how the free parameters are set.
  - (a) is checked against **every notebook row and every run you have made**.  Each observable
    contributes a standardised deviation `z = (observed - predicted) / sd`; the witness passes iff every
    `|z| <= 3` and the sum of `z^2` is below the 0.999 quantile of chi-square with one degree of freedom
    per observable.  A quantity the world fixes *exactly* - a count, a deterministic detector reading -
    must match to six decimals.  Running more experiments therefore constrains the witness more, not less.
  - (b) is checked by recomputing the claim in the witness world.
  - There is no partial credit: a witness that reproduces the evidence but leaves the claim true fails,
    as does one that falsifies the claim but does not reproduce the evidence.  A value outside a declared
    range, a missing parameter or an extra parameter is rejected unread.

## 4. Using the service

    lab spec
    lab run svc=bench batch=64 seq=2048 bits=16 dur=20 seed=0
    lab run svc=bench batch=64 seq=2048 bits=16 draft=small spec_g=4 dur=20 seed=0
    lab run svc=quality batch=64 seq=2048 bits=4 seed=0
    lab run svc=load batch=64 seq=2048 bits=16 rate=12 dur=60 seed=0
    lab run svc=ttc k=16 batch=64 seq=2048 bits=16 seed=0
    lab batch plan.json
    lab status
    lab history

`bench` reports `tokens_per_s`, `ms_per_token`, `kv_bytes_per_seq`, `batch_max`, and -- when speculating --
`accept_by_position` (one estimated rate per proposal position) and their mean `accept_rate`.  `quality`
reports `acc` on a fixed held-out set.  `load`
reports `util`, `p50_ms`, `p99_ms`, or `status="unstable"` when the queue diverges.  `ttc` reports
`cov_at_k` (a correct answer is among the k samples) and `acc_selected` (the verifier's pick is correct).
Cost is accelerator-seconds: a long benchmark, a large-k `ttc` call and a `quality` call at a big batch
cost very different amounts.

## 5. This lab

Settable knobs:
- `svc`: one of bench (default bench)
- `seq`: number in [256, 4096] (default 1024)
- `batch`: number in [1, 1024] (default 32)
- `draft`: one of none, d_lite (default none)
- `spec_g`: number in [0, 4] (default 0)
- `dur`: number in [5, 40] (default 20.0)

Fixed settings (cannot be changed here): `bits`=16

Budget: at most 40 accelerator-seconds per request, 900 accelerator-seconds in total, at most 36 requests.  Charging: accelerator-seconds = measured wall time x replicas (a refused request is not charged).

This lab serves one model at `bits=16` and exposes the decode benchmark only.  The build installed here caps the proposal length at `spec_g=4`; the scheduler that would raise it is not in this build and cannot be run.

`draft=d_lite` is the draft model the team benchmarked.  A `bench` row with `spec_g>0` reports `accept_by_position` (the acceptance rate estimated at each proposal position, in order) and `accept_rate` (their mean).  `ms_per_token` is wall time per **accepted output token**, not per verification step.

`kv_bytes_per_seq` and `batch_max` are exact and describe **this machine**: `batch_max` is the largest batch whose weights, activations and KV cache fit in its memory.  The production deployment in `/app/deploy/serving.yaml` reserves a fixed KV pool of its own, which is a different and smaller number (20.66 GB).

### Known unknowns

- **production p95 context length.** The production replicas serve a traffic mix whose p95 context length lies between 1182 and 1760 tokens; capacity planning uses that p95 length, and the traffic team has not fixed a single figure.  Nothing in this service depends on the choice.  A question about the production replica must cover the whole range.
- **acceptance decay beyond proposal position 4.** The draft model's per-position acceptance decays geometrically, and the decay factor from proposal position 4 on is a *separate* constant from the one governing positions 0 to 3.  It is known only to lie in [0.85, 1.00].  No configuration this build can run exercises a position past 3, so this service behaves identically for every value in that range.
