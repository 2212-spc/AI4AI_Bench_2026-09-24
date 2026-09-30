# EvalLab manual

## 1. What the service models (abstraction boundary)

EvalLab is a **simulated** language-model evaluation service.  No real model is ever called; every
request is answered by a simulator.  The simulator is not a replica of any real evaluation stack.  Its
mechanisms are modelled on regularities reported in the evaluation-methodology literature, but **its
constants were drawn fresh for this lab**.  Published numbers (item difficulties, judge agreement
rates, benchmark accuracies, contamination rates) describe other worlds.  Measure; do not recall.

Objects: a fixed **item bank** of benchmark questions, split into **slices**; a pool of **models**; one
**LLM judge**.  Each item has a difficulty and a discrimination; each model has an ability.  A
`score` request reports how many items were answered correctly; `judge` and `arena` requests report
pairwise preferences of the judge.

The simulator contains the following kinds of effects.  **Any given lab may switch some of them off
(held neutral); which ones are active is not stated.**

- *Item response.*  The chance a model answers an item correctly rises with its ability and falls with the item's difficulty; items differ in how sharply they separate models.
- *Slices.*  A model's ability may differ from slice to slice, and slices may differ in difficulty.
- *Prompt format.*  The same item under a different answer format can give a different result, and the shift can differ by model.
- *Answer extraction.*  Under some formats a model's answer cannot be parsed at all.  Such items are reported as *attempted but not scored*; whether they are counted in a denominator is the analyst's choice, not the service's.
- *Contamination.*  Some items may appear in some models' training data; the resulting advantage grows with exposure and can differ by format.
- *Contamination detection.*  A corpus service can report, per item, either a cheap n-gram **overlap score** or an expensive exact **occurrence count**.  The overlap score is a fixed monotone function of the occurrence count plus item-level detector noise, so it is neither a sound nor a complete test for contamination on any single item.
- *Judge bias.*  A judge's preference can depend on which answer is shown first, on answer length, and on whether the answer comes from its own model family, in addition to answer quality.
- *Ratings.*  Pairwise win rates can be summarised by a Bradley-Terry rating per model; whether the ratings are determined by a set of games depends on which pairs were played.
- *Repetition.*  Repeated scoring of the same model on the same items differs from call to call.  Part of that variation is per-item sampling and part is a shift shared by every item in a call.
- *Selection.*  Nothing in the service selects for you.  If you report the best of several repetitions, that number is not an estimate of the underlying accuracy.

Not modelled: tokenizers, context length, latency, cost of the real API, prompt wording beyond the named formats.

## 2. Guarantees

- **Fixed bank.**  The item bank and its slice labels do not change.  `n` items means a fixed nested
  prefix of one fixed permutation of the split, so two models evaluated at the same `split`/`n` see the
  **same items**, and `n` = 200 is a subset of `n` = 400.
- **Fixed laws.**  Every mechanism is a fixed law whose constants do not depend on which request you
  make.  There are no hidden thresholds, **except** where Section 5 lists a *known unknown*.
- **Determinism.**  An identical request with the same seed returns identical numbers (and is charged
  again).  Different seeds give independent draws.
- **Notebook rows are real service responses.**  The team's notes that accompany them are the team's
  interpretation and may be wrong.

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
- **Verdicts** take `{"verdict": "supported" | "refuted" | "undetermined"}`.  *supported*: true in every
  world consistent with what can be known; *refuted*: false in every such world; *undetermined*: true
  in some and false in others (i.e. it hinges on a known unknown).

## 4. Using the service

    lab spec                                              # knobs, fixed settings, caps (free)
    lab run svc=score model=M1 split=all fmt=mc_letter n=200 reps=1 seed=0
    lab run svc=judge a=M1 b=M2 split=all n=150 order=both seed=1
    lab run svc=arena a=M1 b=M3 games=200 seed=2
    lab run svc=corpus mode=fast n=600 seed=0                 # cheap overlap score for every item
    lab run svc=corpus mode=scan n=60 seed=0                  # exact occurrence counts (expensive)
    lab batch plan.json                                   # a JSON list of requests, run in order
    lab status                                            # credits used / left (free)
    lab history                                           # every request you have made (free)

`score` reports `n_items` (items presented), `n_scored` (items whose answer could be extracted),
`acc` (mean over repetitions of correct / n_scored), `reps_acc` (one accuracy per repetition) and
`by_slice` counts for the first repetition.  Add `detail=1` for per-item outcomes.
`judge` reports win counts, and `by_order` separately for each presentation order.
`corpus` reports one row per item: `overlap` in [0,1] under `mode=fast`, or the exact `occurrences`
count under `mode=scan`.  Both modes address the same fixed corpus and are deterministic: re-asking
returns the same numbers, so repetition buys nothing.  Items are named by index and item indices are
stable across every service, so `corpus` rows can be joined to `score detail=1` rows.
A request above the per-request cap, beyond the request limit, or beyond the remaining budget is
refused and not charged.  Results are also appended to `/app/lab_runs.jsonl`.

## 5. This lab

Settable knobs:
- `svc`: one of score, corpus (default score)
- `model`: one of M_new, M_ref (default M_new)
- `split`: one of all, s0, s1, s2 (default all)
- `fmt`: one of mc_letter, mc_cloze (default mc_letter)
- `n`: number in [10, 1500]
- `reps`: number in [1, 400] (default 1)
- `detail`: one of 0, 1 (default 0)
- `mode`: one of fast, scan (default fast)

Budget: at most 8e+03 eval credits per request, 3e+04 eval credits in total, at most 36 requests.  Charging: credits = items x repetitions x per-call cost of the model(s) involved.

Models: `M_new` is the release candidate, `M_ref` the reference the memo compares it against.  Both cost 0.05 credits per item per repetition.

Formats: `mc_letter` presents the options and reads back a letter; `mc_cloze` scores the options by likelihood.  An item whose answer cannot be extracted is reported as presented (`n_items`) but not scored (`n_scored`); the reported `acc` is correct / `n_scored`.

**The release harness that will publish these numbers counts an item whose answer cannot be extracted as wrong, over every item presented.**  The questions below say which convention they mean; the service's own `acc` is not the release number.

`svc=corpus` addresses the pretraining corpus of the models, not the item bank: `mode=fast` returns a cheap n-gram `overlap` score per item (0.03 credits per item), `mode=scan` the exact `occurrences` count (1.2 credits per item).  Both are deterministic, and both join to a `score` row's `detail` table by item index.

### Known unknowns

- **release weight on slice 0.** The release report aggregates the three slices with weight w0 on slice 0 and the remaining 1 - w0 split 2:1 between slice 1 and slice 2.  The product team will fix w0 somewhere in [0.15, 0.85] and has not done so.  Nothing in this service depends on that choice.  Questions about the release report must cover the whole range.
