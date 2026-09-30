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
- **Audit questions** ship a teammate's analysis script under `/app/analysis/`.  It reads the logs in
  `/app/notebook/`, and the number it printed is the one quoted in the team's notes.  Exactly one
  **silent inferential defect** has been introduced: the script runs to completion and prints a
  plausible number, and the number is wrong.
  Answer `{"defect": "<id>", "site": "<file>:<line>", "corrected": {"lo": x, "hi": y}}`.
  - `defect` is one id from the taxonomy in Section 6.
  - `site` is the line where the wrong inference is made, not a line that merely prints or returns the
    result.  A small span around that line is accepted.
  - `corrected` is an interval containing what the analysis gives once the defect is fixed and nothing
    else is changed; the question states how wide it may be.  Recompute it from the shipped logs - the
    logs are the team's real data, and the lab is not needed for this part.
  - All three parts must be right.  The script also contains choices that are merely ugly, redundant or
    unusual; those are not the defect unless fixing them moves the reported number.

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

Budget: at most 4e+03 eval credits per request, 6e+03 eval credits in total, at most 12 requests.  Charging: credits = items x repetitions x per-call cost of the model(s) involved.

The team's own passes are in `notebook/runs.jsonl` (the per-item tables the memo cites are there in full), the exact corpus scan in `notebook/scan_exact.csv`, the cheap detector's scores in `notebook/detector_fast.csv`, and the script that produced the memo's headline in `analysis/headline.py`.

A `score` request with `detail=1` returns a per-item table (`item`, `slice`, `scored`, `correct`) for its first repetition, which is what the notebook's passes recorded.

### Known unknowns

- **release weight on slice 0.** The release report aggregates the three slices with weight w0 on slice 0 and the remaining 1 - w0 split 2:1 between slices 1 and 2.  Product has not fixed w0; it is known only to lie in [0.15, 0.85], and no request to this service depends on that choice. Questions about the release report must cover the whole range.

## 6. Defect taxonomy

An audited analysis contains **exactly one** defect from this list. Every defect here is *silent*: the script runs to completion and prints a plausible number. Name the id, not a description of your own.

- **`D1_selection_max_over_seeds`** - best-of-several reported as an estimate. The script runs a measurement several times (seeds, repetitions, sampling temperatures) and reports the best of them as if it were an estimate of the underlying quantity. The expected maximum of k draws exceeds the mean by roughly the noise scale times a factor growing with k, so the reported number is biased upwards by an amount that depends on how many times it looked.
- **`D2_denominator_scored_only`** - unparseable answers dropped from the denominator. Items whose answer could not be extracted are excluded from the denominator instead of being counted as failures (or reported separately). Accuracy is then computed over the items the extractor happened to succeed on, which is not the set the claim is about, and the size of the error differs by model and by answer format.
- **`D3_aggregation_weight_mismatch`** - per-slice means averaged with the wrong weights. A number reported as an overall rate is computed as the unweighted mean of per-slice rates (macro) where the claim requires the item-weighted rate (micro), or the reverse. The two differ whenever the slices differ in size, and they can move in opposite directions when slice difficulty is correlated with slice size.
- **`D4_position_bias_uncorrected`** - judge preferences pooled over one presentation order. Pairwise judgements are collected with the two answers in a fixed order, or in both orders but pooled without checking the order effect, and the resulting win rate is reported as a quality comparison. The judge's preference for whichever answer it sees first is then added to the quality signal.
- **`D5_paired_test_as_unpaired`** - a paired design analysed as if unpaired. Two systems are measured on the *same* items, but the comparison is made with the unpaired standard error (or two independent confidence intervals are checked for overlap). Item difficulty is shared variance that a paired analysis removes, so the reported uncertainty is far too large and a real difference is called undetermined.
- **`D6_ratio_of_means_vs_mean_of_ratios`** - ratio of means reported as the mean of ratios. A per-unit quantity (cost per token, tokens per second, speed-up per request) is computed by dividing one total by another, where the claim is about the typical per-unit value; or the reverse. The two agree only when the denominators are equal across units, so the reported number is silently dominated by the largest unit.
- **`D7_contaminated_items_included`** - items known to be contaminated left in the reported set. The analysis has the evidence that some items are in a model's training data (a detector score, an occurrence count) and reports the headline number over all items anyway, or filters with a threshold on a noisy detector and reports the result as clean. The excess on the affected items is then attributed to ability.
- **`D8_threshold_on_noisy_max`** - a threshold crossing declared from a noisy statistic. The script declares that a quantity has crossed a threshold (an emergence point, a regression, a stability limit) from a single noisy measurement, or from the first of several measurements to cross it. The crossing point of a noisy series is biased towards wherever the series is noisiest.
- **`D9_survivorship_dropped_diverged`** - failed runs dropped from a comparison. Runs that diverged, timed out, were refused or returned nothing are dropped from the aggregate, and the mean over the survivors is reported as the configuration's outcome. Failure is not independent of the setting under study, so the setting with the most failures looks best.

Anything else you dislike about the script (naming, redundant work, a hard-coded path, a loop that could be vectorised, a comment that contradicts the code) is not a defect unless fixing it changes the reported number.

