# ScaleLab manual

ScaleLab is a **simulated** language-model pretraining lab.  You cannot train real models here; every
`lab run` is answered by a simulator.  The simulator is not a replica of any real training stack.  Its
mechanisms are modelled on regularities reported in the scaling-law and training-dynamics literature,
but **its constants were drawn fresh for this lab**.  Published numbers (Chinchilla exponents,
"optimal" learning rates, epoch rules of thumb, and so on) describe other worlds.  Measure; do not recall.

## 1. What the simulator models (abstraction boundary)

Quantities: N = non-embedding parameters; D = training tokens; B = batch size in tokens;
lr = peak learning rate; FLOPs = 6*N*D (a run is charged this even if you only look at its end).
`loss` is the final validation loss in nats/token on a fixed held-out web set.

The simulator contains the following kinds of effects.  **Any given lab may switch some of them off
(held neutral); which ones are active is not stated.**

- *Size and data.*  Loss falls with N and with the effective number of tokens, with diminishing returns, towards an irreducible floor.
- *Learning rate.*  Loss rises on both sides of an optimal peak learning rate; the rise need not be symmetric.  The optimum may move with N, D and B.
- *Batch size.*  Beyond a critical batch size, larger batches waste tokens; the critical batch size may move with the data horizon.
- *Stability.*  Above an edge learning rate a run diverges.  The edge may move with N, warmup and qk-layernorm.  Divergence is detected early; a diverged run is charged 10% of its FLOPs.
- *Weight decay.*  AdamW's averaging timescale B/(lr*wd*D) has an optimum that may depend on tokens per parameter.
- *Repetition.*  Repeating unique tokens (epochs) gives diminishing value.
- *Quality filtering.*  Filtering can raise the value of each token but shrinks the pool of unique tokens.
- *Metrics.*  Task metrics (accuracy, exact match) are functions of the model's loss and of the metric's definition; some are not linear in loss.
- *Contamination.*  Some data sources can contain benchmark items; memorisation can grow with scale and exposure.
- *Schedules.*  An intermediate checkpoint of a long run is not the same as a finished short run; the gap depends on the learning-rate schedule.
- *Seed noise.*  Every metric has seed noise; its size can depend on N.

Not modelled: architecture beyond N, tokenizer, hardware, data order, wall-clock time.

## 2. Guarantees

- **Scale consistency.**  Every mechanism is a fixed law whose constants do not depend on scale.
  Relationships you measure in the lab continue, by the same law and with the same constants, to the
  production scales named in the questions.  No effect switches on or off outside the lab's range,
  and there are no hidden thresholds, **except** where Section 5 lists a *known unknown*.
- **Determinism.**  Re-running an identical configuration with the same seed returns the identical
  numbers (and is charged again).  To average out noise, use different seeds.
- **Notebook runs are real lab runs** (seed 0 unless stated), produced by the same simulator you query.
  The lab notes that accompany them are the team's interpretation and may be wrong.

## 3. How answers are graded

- **Numeric questions** take an interval `{"lo": x, "hi": y}`.  The interval must be the *set of values
  consistent with everything that can be known*: what the lab can measure within your budget, plus the
  guarantees and documented ranges in this manual.
  - If the lab pins a quantity down, answer a degenerate interval (`lo == hi` = your estimate).  Seed
    noise is handled by the grading tolerance, so **do not** widen for statistical uncertainty.
  - If the quantity depends on a known unknown (Section 5), `lo`/`hi` are the minimum and maximum over
    its documented range.
  - Each endpoint is compared with the true endpoint.  The tolerance is calibrated to how precisely a
    well-designed set of experiments within the budget can pin that endpoint down.
- **Verdicts** take `{"verdict": "supported" | "refuted" | "undetermined"}`.  *supported*: true in every
  world consistent with what can be known; *refuted*: false in every such world; *undetermined*: true
  in some and false in others (i.e. it hinges on a known unknown).
- **Decisions** take `{"choice": "<option>"}`: pick the option with the best outcome as defined in the question.

## 4. Using the lab

The `lab` command talks to the lab service (no other network access is needed or available).

    lab spec                                  # knobs, fixed settings, caps, metrics (free)
    lab run N=6e7 D=1.5e10 lr=0.002 seed=3    # one training run; prints a JSON result
    lab batch plan.json                       # a JSON list of runs, e.g. [{"N":6e7,"D":1.5e10,"lr":0.002,"seed":3}, ...]
    lab status                                # compute used / left (free)
    lab history                               # every run you have made (free)

Settable knobs must all be given (unless the spec lists a default); fixed settings cannot be changed.
A run above the per-run FLOP cap, beyond the run limit, or beyond the remaining budget is refused and
not charged.  `lab batch` stops at the first refused run.  Results are also appended to
`/app/lab_runs.jsonl`.

## 5. This lab

Settable knobs:
- `N`: number in [1e+07, 3e+08]
- `D`: number in [2e+08, 1e+11]
- `q`: number in [0, 0.9] (default 0.0)
- `sub`: number in [1e+08, 2e+11] (default 200000000000.0)

Fixed settings (cannot be changed here): `B`=5e+05, `wd`=0.1, `wu`=0.01

Compute: at most 6e+18 FLOPs per run, 8e+19 FLOPs in total, at most 40 runs.
Metrics reported by `lab run`: loss.

Data knobs: `q` is the fraction of the raw corpus removed by the quality classifier (lowest-scoring documents first).  `sub` is the number of raw unique tokens sampled from the corpus *before* filtering (an ablation subset); the filter then keeps a fraction 1-q of it.  Default: the whole current corpus, 2e11 raw unique tokens.  A run whose D exceeds its unique tokens sub*(1-q) repeats data (several epochs).  Every run uses a learning rate tuned by the lab for its configuration, so there is no lr knob.

### Known unknowns

- **next corpus snapshot size.** Production trains on the next snapshot of the same corpus (same quality distribution; the classifier and its q scale are unchanged).  Its deduplication is still running: it will hold between 2e10 and 1e11 raw unique tokens, and nothing in this lab depends on the exact number.  Questions that name a snapshot size use that size; questions about the snapshot 'as it will be' must cover the whole range.
