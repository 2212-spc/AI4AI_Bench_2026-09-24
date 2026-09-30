# Runs excluded from the results matrix

| run prefix | why |
|---|---|
| `k6w1__*`, `k6v1__*` | launched against `tasks/k6-ramp-w1/v1`, built before the K6 redesign (per-run corpus switch + flat ramp). Their stored params are no longer the params the module simulates, so their scores measure nothing. Superseded by `k6f1__*`. |
| `k3w1__*` | `WIDTH_CAP=0.40`, superseded by `k3v1__*` (cap 0.20). Kept only as the evidence for the cap change (gpt-6 passed with a +-0.19 interval). |
| `k4w1__*` | first K4 instance, superseded by `k4-tts-ws1`. Kept as the evidence for the K4 difficulty fix. |

## k3w1__* (3 runs) - stale task, not a model result

`tasks/k3-edge-w1` was built and run while `k3_edge.WIDTH_CAP` was **0.40**; its instruction says
"Width at most 0.40" in two places. The cap was later measured down to **0.20** (the oracle's own forecast
error is p90=0.066), and because `l15.grade` imports the *live* module while the instruction is frozen at
build time, re-grading those runs applied a bar the agents were never shown:

| run | first grade | re-grade under 0.20 |
|---|---|---|
| k3w1__gpt6__r1 | PASS 0.9992 | fail 0.0 `R0_artifact: interval width 0.3800 exceeds the cap 0.20` |
| k3w1__fable__r1 | fail 0.0 `R0_artifact` | fail 0.0 |
| k3w1__haiku45__r1 | fail 0.889 | fail 0.0 |

gpt-6 submitted a width-0.38 interval, which was **legal** under the task it was given. Reporting that as a
frontier failure would be reporting a bug in the harness as a result about the model.

Fixed at the source rather than by hand: `l15.build` now stamps every module constant into
`hidden/grading_consts.json`, and `l15.grade` raises `StaleTask` when the live module disagrees, so a
re-tuned constant surfaces as an integrity error in the sweep instead of as a silent re-grade. The two
`k3-edge-w*` tasks carry their real build-time cap (0.40) in that file, so they now fail the check loudly.
Superseded by `k3-edge-v1` / `k3-edge-v2`, which are consistent at 0.20.

## k1b1__* / k1b2__* - superseded, mechanism changed mid-sweep

Built from `k1-judge-b1/b2` (seeds 5 and 19) while the K1 judge had error rates of 0.06-0.16 and the oracle
used the per-stratum **inversion** `w = (jm - fp)/(1 - fp - fn)`.

`k1b2__haiku45__r1` **passed at 1.00 coverage / 0.104 width while ignoring the judge entirely** - it bought
220 stratified human labels and averaged them under the production mix. That is unbiased by construction,
and measurement confirmed the judge was worth nothing at that budget: judge-free width 0.102 vs calibrated
0.129. The weakest model in the panel passing by discarding the task's central object is a design failure,
not a model result.

Two things were wrong and both were found by that run:
1. the judge disagreed with humans ~20% of the time, so its 900 free verdicts carried almost no information
   beyond what 220 labels already gave;
2. neither the oracle nor the baseline applied the **finite-population correction** the estimand calls for
   (the target is *this arena's* realised rate), which hid the judge's advantage entirely.

Rebuilt as `k1-judge-c1` (seed 11) / `k1-judge-c2` (seed 14): the judge is accurate (base error 0.02-0.05)
but still per-stratum asymmetric, and the oracle is the **rectified / prediction-powered** estimator
`w_s = jm_s + E[y - judge]` with FPC and a pooled disagreement variance. Measured separation now:
oracle 0.025 width at 0.96 coverage; judge-free 0.053; no-FPC 0.056; inversion 0.107 - against a cap of
0.040. The judge is now necessary.

## k2-mix-* : R1_mixture_quality does not identify the instance (measured 2026-09-28)

Not excluded - the runs are valid and the grade is correct.  But R1 is close to free, and the report
must say so rather than counting it as evidence of experimental skill.

Found while reading `k2w1__gemflash__r1` (PASS 0.9985, regret 0.0001).

**CORRECTION (same day, from an independent audit).**  This entry first claimed gemflash had passed on
four `train` calls at a single design point, i.e. by recalling a published mixture rather than measuring.
That was WRONG, and the error was mine: I read `app/lab_log.jsonl`, which `l15/cli.py` writes only for
calls made through the `lab` CLI.  gemflash made 4 CLI calls and then switched to calling the lab
programmatically.  The authoritative ledger, `<run>/lab_ledger.jsonl`, has **69 train calls over 29
distinct mixtures and 9 distinct (N,D) points**, sweeping N over 8x (5e7 -> 4e8) and D over 8x
(1e9 -> 8e9).  That is a real scaling sweep, and `grade.json.lab_calls` said 69 all along.  gemflash
earned this pass.  The under-count affects 7 runs, all gemflash, by up to 366x (k3v1: 733 real vs 2
logged) - any analysis reading app/lab_log.jsonl is wrong about gemflash effort.

The R1 defect below is nonetheless real: it was confirmed by direct measurement (`explore/k2a.py` and the
transfer gate), not by the mis-read ledger.  What changes is the story: R1 being free is a property of
the TASK, not evidence that this model cheated.

Measured, `explore/k2a.py` - replaying that one submission against every gated instance:

  seed   regret   tol      covers   verdict
  3      0.0015   0.0087   False    fail
  8      0.0001   0.0082   True     PASS      <- the instance it was produced on
  10     0.0024   0.0095   False    fail
  16     0.0015   0.0082   True     PASS

R1 (regret <= tol) passes on 4/4; only R2 (the forecast interval) ever fails.  The reason is that the
optimum hardly moves across instances - per-domain spread of r_opt over the four gated seeds is
[0.057, 0.083, 0.012, 0.052] around a mean of [0.554, 0.285, 0.027, 0.134].  One fixed mixture sits
inside tol everywhere.  `strat_recite` was supposed to catch exactly this and does not: it recites a
wrong *repetition* constant (R*=15), which is a different error from reciting the right mixture.

Consequences:
  - K2's difficulty lives entirely in R2_forecast_covers (the data-constrained scaling extrapolation),
    which is also where gpt-6 failed: it submitted [2.19, 2.24], its own best proxy composite (2.2177
    at N=1e8/D=2e10, under the byte-identical mixture it went on to submit), as the forecast for a run
    25x larger - M1, validity-domain extrapolation, in its purest form.  gemflash passing and gpt-6
    failing the same task is a REAL inversion on this instance: the cheap model ran the wider sweep
    (69 calls, 9 scale points) and the frontier model ran 16 calls and reported a proxy number as a
    target forecast.  gpt-6 passed on its second attempt (r2, 66 calls) - so the difference tracks
    whether it did the sweep, not whether it could.
  - The fix is a gate, not a tolerance: a `fixed_mixture` decoy (one mixture replayed across instances)
    must fail, which forces sample_params to spread the optimum far enough that r_opt is instance-
    identifying.  Deferred to v8 - noted here rather than silently shipped.

## harness bug found 2026-09-28: the slice guard could exceed the slice (SLICE_NEED_CAP)

Not an exclusion either - but six runs were frozen by it, and without the fix they would have been
written up as "frontier models that never finished", which is a harness result wearing a capability
result's clothes.  Third such defect this phase (after the stale grading constants and drive.py
starvation), and the same shape every time: the harness failed in a way that LOOKED like data.

`gpt_agent.tick` refuses to start a model call it does not expect to finish inside the host slice
(a timed-out call is lost work).  The guard:

    need = min(SLICE_NEED_CAP, max(75, 1.3 * max(recent_lat[-5:]) + 15))
    if rem < need: return

With SLICE_NEED_CAP=140 and a recent max latency of 72.1 s, `need` = 108.7 s, while the driver's 95 s
slice offers ~85 s after startup.  So `rem < need` on EVERY tick and the run never calls again.  Six
runs (k8a2 gpt6, k6f1 gpt55, k6f1 gpt6, k6f2 gpt55, k8a2 gpt55, k1c1 gpt6) sat at frozen n_calls
through ~15 consecutive ticks while `state.json` kept being rewritten - so they looked alive in the
status line and were idle in fact.

Fixed by capping at 70 s (must be comfortably under the 120 s per-bash-call ceiling this harness runs
under).  The comment in the code already warned about exactly this failure - it was written after an
identical livelock on 2026-09-25 - but the cap was left at a value from an era of longer slices.  A
guard that can exceed the thing it guards is not a guard.

Immediate effect: the very next tick advanced all six and three finished, including the two results
that turned out to matter most (k6f1 gpt6 and gpt55 both scoring BELOW the do-nothing baseline).

---

## k8a1: the format example was a working answer (leak, verified, fixed)

**Not an exclusion — a defect in the task, found by an independent audit and confirmed by measurement.**

`k8_post.docs()` shipped `docs/report_format.md` containing a concrete worked example:

    { "cause": "capacity", "fix": {"capacity": 0.74, "lr_scale": 1.2} }

On **k8-post-a1 (seed 1) the true cause IS `capacity`**, so the example named the right answer. How much
that was worth, measured by adding a `copy_doc` strategy that reads the doc and submits it, spending nothing:

| doc version | copy_doc result |
|---|---|
| old (worked example) | **pass=True, score 0.9394, all of R0–R3 green, 0.0% of budget spent** |
| new (placeholders)   | pass=False, score 0.0, fails R0 |

So the old instance could be passed outright without one lab call. The submissions agree:

| run | cause | fix | pass |
|---|---|---|---|
| k8a1 gemflash | capacity | `{capacity: 0.773, lr_scale: 1.2}` | True |
| k8a1 gpt6 | capacity | `{capacity: 0.773, lr_scale: 1.2}` | True |
| k8a1 sonnet5 | capacity | `{capacity: 0.773, lr_scale: 1.2}` | True |
| k8a1 gpt55 | **lr_scale** | `{capacity: 0.773, lr_scale: 1.2}` | False (R2 only) |
| k8a1 fable | capacity | `{capacity: 0.773, lr_scale: **1.15**}` | True |
| k8a1 haiku45 | capacity | `{capacity: 0.773}` | False |

Four of five models submitted the doc's `1.2` byte-for-byte; only fable deviated. `0.773` is the cap stated
in the instruction, so the whole fix vector was readable off the materials. The clearest evidence that R1
was not earned: **gpt55 scored 1.008 on R1 while declaring the wrong cause** — the fix recovered full
quality without the diagnosis being right.

**Why the gate did not catch it.** Every `fail` strategy in k8_post played a bad *reasoning process*
(blame the leak, fix everything, take the biggest offline mover). None played the *documentation*. A gate
only tests what a strategy plays, and the materials a task hands the agent are part of its attack surface.

**Fixes, all at the source:**
1. `docs/report_format.md` now ships non-instantiable placeholders (`"<knob>": 0.0`), so a model that
   copies it fails R0 loudly instead of passing quietly.
2. `strat_copy_doc` added to `k8_post.STRATEGIES` with `expect="fail"` — the gate now fails if the doc is
   ever made copyable again. This is the general lesson, not a k8 patch.
3. Swept every gated family for the same defect (doc numbers vs. truth scalars, rel tol 5%): k1, k4, k6
   clean; k3's hits are the published allowed-width list, i.e. the answer *space* (`eta_opt`, the part
   that must be found, appears in no doc); k2's single hit is a coincidental collision with a decoy
   constant. **k8 was the only real one.**

**How to read k8a1's 4/6.** It is not a capability measurement. A fresh instance `k8-post-a3` (same seed 1,
new salt, corrected doc) was built and re-run on all six models; use those numbers instead. k8a1 is retained
only as the evidence for this entry.

**Also fixed:** `k8_post.grade()` returned numpy bools, which serialised into grade.json as `1.0`/`0.0` and
forced every reader to special-case `v["ok"] is True or v["ok"] == 1.0`. Now coerced with `bool()` at the
source.

**Fourth integrity defect this phase, and again the same shape:** the pipeline produced a number that
looked like a model result and was actually an artifact of how the task was built.

---

## k8: the instruction named the cause (the real defect; the doc leak was a symptom)

**Supersedes the "How to read k8a1's 4/6" line in the previous entry.** Fixing the doc raised the bar but
did not fix the task: k8a3 (same seed, corrected doc) scored **2/6** instead of 4/6, and the remaining
passes were still not measuring diagnosis.

**What was wrong.** `instance_truth` set `release_knob = p["cause"]`, and the instruction printed it in
bold in its second paragraph: *"the release changed **`capacity`** from 1.000 to 0.676"*. The release
moved exactly one knob, and that knob was always the answer — **20/20 seeds**. Measured with a strategy
(`strat_read_instruction`) that does no attribution at all, just reads the knob off the instruction and
re-tunes around it:

| | result |
|---|---|
| read-the-instruction, zero diagnosis | **passed 5/6** (seeds 1/6/8 × 2 salts) |

The task was scoring the re-tuning search, not the attribution it claims to test. Every `fail` strategy
played a bad *reasoning process*; none played the *materials*. Both leaks — doc and instruction — were
invisible for the same reason.

**The fix, and what it cost.** The release now moves **all three** knobs by nearly the same relative
amount, and the cause is *derived* from the resulting quality surface rather than drawn in advance. That
required rebuilding the instance generator, because each intermediate version leaked differently:

1. *Decoys moved slightly.* → magnitude still ranked the causes; blame-the-biggest-mover was the answer.
2. *All three moved equally, drawn independently.* → biggest-mover still hit **91/120 (76%)** vs 33%
   chance. Independent draws make magnitude informative.
3. *One shared magnitude ±8%.* → biggest-mover fell to **43%**, but with uniform sensitivities no knob
   dominated (median margin 0.084) and the gate accepted **2/150** seeds.
4. *Equal moves, unequal hidden sensitivity.* → one knob per instance is drawn sensitive. This is the
   design: **the release report shows three equal-looking changes, and which one mattered is discoverable
   only by intervening.**

Finding the right sensitivity lever took measurement rather than guessing. For `lr_scale` the obvious
constant (`lr_c`) *saturates* — stuck-cost peaked at 0.079 and then fell as it rose, because stability is
`1/(1+lr_c·d²)`. The lever is `lr_beta`, how strongly the best lr tracks capacity, i.e. how **compensable**
a stuck lr is: 0.50 → 0.10 took the same instance from 0.076 to 0.345. For `capacity` it is `ceil_k`, not
`ceil_b` (which dragged lr_scale up with it). `cov_g` is inert (identical stuck-costs over 0.2–1.4).

**Two rubric items had to be restated**, because under coupling "resetting one knob" stopped meaning
anything — resetting the true cause alone often makes quality *worse* (−0.47 on the normalised scale), and
the old R2 rejected even the oracle 0/4. Both now use the same definition: *how much does leaving this
knob where the release put it cost, once the other two are re-tuned freely*, and the cause must rank first
by ≥15 points. The knobs are coupled on purpose, so the rubric has to speak the language of the coupling.

**Cause balance is itself a gate condition.** A pool that is 86/120 mixture would let "always answer
mixture" score without diagnosing. `cov_a` alone decides this (0.44 → mixture 70/80; 0.15 → 8/80).

**Three permanent gate strategies** now hold these properties, so a regression fails the gate instead of
reaching a reader of the results: `copy_doc` (the docs), `read_instruction` (the release report), and the
per-instance screen rejecting any seed where the biggest mover *is* the cause. Near-chance in aggregate
is not enough — on a specific shipped instance the heuristic is either right or wrong, and if it is right
that instance is free.

**Transfer-gate fix found along the way.** `R3_fix_acts_on_cause` transferred 8/8 and was flagged as a
leak, but it compares the fix to the cause *the same artifact declares* — self-consistency, which can
never do anything but transfer. Tasks now declare `World.SELF_CONSISTENT` instead of the gate inferring
exemption from an `R0` name prefix. A naming convention is not a specification.

**Status.** k8a1 and k8a3 are retained as evidence, not as capability measurements. Two fresh instances
with different causes — `k8-post-b1` (seed 19, cause `lr_scale`) and `k8-post-b2` (seed 27, cause
`mixture`) — pass the full gate (oracle 4/4 and 3/4; every decoy, shortcut and nop fails) and are the ones
to report.

---

## k2: the answer barely moved between instances (the open defect, now closed)

**Symptom.** `transfer_gate` reported `R1_mixture_quality` transferring **7/9**: one instance's
`mixture.json`, replayed on instances it had never explored, kept satisfying the quality item. The
earlier `k2w1__gemflash__r1` run had already shown what that buys — regret 0.0001 after four `train`
calls that were all the same uniform mixture at the same tiny scale, i.e. four repeats of one design
point, which measure run-to-run noise and cannot identify a mixture.

**Measurement.** Searching the whole simplex against the pool for the single best FIXED mixture:

| pool | best fixed mixture | clears R1 on |
|---|---|---|
| seeds 1–40 (all) | web .44 / code .40 / math .06 / papers .10 | **40 / 40** |
| 19 gated seeds | web .38 / code .28 / math .10 / papers .24 | **18 / 19** |

and it did not take the pool-optimal mixture to do it. Five mixtures a practitioner can write down
before touching the lab cleared R1 as follows: `quality_heavy` 17/19, `even_ish` 14/19, `code_heavy`
13/19, `llama_ish` 11/19. A model that had merely seen a sensible pre-training mixture scored like one
that had run the experiment.

**Root cause — the draws were too narrow, and no single one of them was the problem.** Best-fixed-mixture
over 24 seeds, widening one group of constants at a time: `U` 20/24, `beta` 23/24, `W` 23/24, `lam`
21/24, **all four together 10/24**. The optimum is set by where each domain runs out of unique text (U)
against how fast its eval component still improves (beta), routed through transfer (W) and then weighted
(lam); narrowing any one pins the optimum back down through the others. `lam` had been the constant
`[0.34, 0.33, 0.33]` on every instance.

**The bar was measuring the reference answer, not the task.** With `TAU = 0.80` the prior-knowledge
mixtures cleared 11–18 of 19; at `TAU = 0.94` they clear at most 7 of 19. The reason `TAU` could not
simply be raised before is that the oracle itself only scored 0.77–0.93 — its own search was the binding
constraint. Two measurements settled that:

* a **third** response-surface refinement round made the oracle *worse* (0.89→0.87, 0.77→0.64): a
  quadratic/exponential surrogate fitted to noisy points has nothing left to learn from a tighter cluster;
* fitting the **mechanism family** instead — scaling law plus repetition decay, with the published corpus
  sizes used to place each run in epochs — scored 0.96 / 0.999 / 0.997 / 0.998 / 0.992 on the same seeds
  and salt, for the same budget.

So the old oracle was demoted to a `fail` strategy (`oracle_rsm`) rather than deleted. It does everything
the task asks conceptually — it reproduces the target's repetition regime at proxy scale — and still lands
5–25 points short, so "had the right idea, fitted a generic surface" is no longer a pass, and if a future
edit makes the task easy enough for the blind surrogate again, the gate fails loudly.

**A fix that was tried and reverted.** Forecasting the target loss from the same fitted mechanism looked
obviously right and was wrong: biased **+0.23 to +0.60** on 8 of 10 (seed, salt) pairs, against +0.001 to
+0.010 for the separate N/D sweep. The mechanism fit is good at *ranking* mixtures and bad at absolute
level, because W, B and lam trade off — a fit can get every comparison right with the whole surface
shifted. Choosing the mixture and forecasting its loss are different estimation problems on the same
data, which is why the task asks for both.

**`WIDTH_CAP` was chosen, not measured.** Against the oracle's own forecast error over 20 (seed, salt)
pairs — median 0.009, p75 0.017, p90 0.045 — the old cap of 0.050 allowed a ±0.025 interval and R2 failed
the *reference answer* on 25% of pairs. Raised to 0.090, still far below the myopic-to-optimal gap
(0.036–0.088 on the shipped instances), so an interval wide enough to cover by default is still not wide
enough to be uninformative. `cheat_wide` remains a gate.

**New gate machinery.** `pool_gate()` — the same lesson as k8's, in a second family: whether the answer
moves between instances is a property of the POOL and a per-instance screen is structurally unable to see
it. Its bar is not a tolerance around chance, because `best_fixed` can never fall below `1/n` (each
instance's own optimum clears itself); it is the crisp property **no single mixture serves two
instances**, which cannot be met by shrinking n. The prior menu is held to the same bar.
`diverse_pool()` selects seeds whose optima are far apart, at selection time, deterministically.

**Result.** Of 400 seeds, 34 pass the per-instance screen. The shipped four — **8, 41, 99, 374** — each
pass the full strategy gate (oracle 4/4, 4/4, 4/4, 3/4; `oracle_rsm`, `myopic`, `recite`, `natural`,
`cheat_wide`, `nop` all fail; `prior_mixture` at or below its guess rate), have optima at least 0.445
apart in L1, and pass `pool_gate` at its floor (best_fixed 0.25 = 1/4, worst prior 0.25). Transfer:
**`R1_mixture_quality` 0/9**, down from 7/9. Seeds 27, 117, 164 and 253 were rejected by the gate, not by
hand: 164's oracle is unstable across salts (2/4), the other three let `oracle_rsm` through on one salt.

**Superseded.** `k2-mix-ws1` (seed 8) and `k2-mix-ws2` (seed 10) were built under `TAU = 0.80` and
`WIDTH_CAP = 0.050`; their frozen `grading_consts.json` no longer matches the module, which is the stale-
task condition the build-time constant stamp exists to catch. They are retired in favour of
`k2-mix-d8 / d41 / d99 / d374`. Any k2 number from the ws instances is a measurement of the old bar.

## The `api_errors >= 5` hold-out rule: added, measured, and reverted (2026-09-28)

Between the two audits above I added a third hold-out rule to `l15/grade.py`: a run whose `api_errors`
counter reached 5 was marked `truncated` and dropped from the headline tally. The motivating observation
was real - `k6f2__gpt55__r1` reported `status: done` with 11 api_errors and only 3 lab calls, and was
being counted as a considered failure. The inference from it was wrong, and the rule is now gone.

`api_errors` counts RETRIED TRANSIENTS, not lost turns. `harness/gpt_agent.py:161` increments it in the
`except` around one API call, saves, and returns; the drive loop calls straight back in. The harness only
abandons a run at `> 40`, and when it does it writes `status: "api_failed"` - which is what the status
test should have covered, and now does. So `done` with a nonzero count means the retries were RECOVERED
FROM: the model kept going and then chose to stop.

What the rule actually did. First measured over the 44 runs that carried the field at the time (9 at
>= 5, 3 of them passes); after re-grading every run under one consistent rule the true figures are **129
runs carrying the field, 15 at >= 5, of which 7 are PASSES**. The first numbers are left visible here
because they are how the defect was found, and because a count taken mid-re-grade is exactly the kind of
stale measurement this file exists to record:

| run | api_errors | model calls | lab calls | spent | verdict it was hiding |
|---|---|---|---|---|---|
| k2d99__gemflash__r1 | 8 | 144 | 130 | 1.00 | **PASS** - full scaling-law write-up |
| k1c1__gpt6__r1 | 5 | 12 | 5 | 1.00 | **PASS** |
| k4w2__gemflash__r1 | 11 | - | 11 | 0.41 | **PASS** |
| k6f2__gpt55__r1 | 11 | 24 | 3 | 0.99 | fail R1_time - py_compile-validated sched.py |
| k2d374__gemflash__r1 | 11 | - | 118 | 1.00 | fail R2 (center err 0.57) |

Seven of the fifteen were PASSES. The rule was suppressing evidence in both directions at once, and the
motivating run turns out to be a genuine considered failure: 24 model calls, a shipped `sched.py` that
it validated with `py_compile` against a fake environment, and a real R1_time miss. Its 3 lab calls were
all spent on PROFILING (`profile_seconds` 8.22, the highest in the k6 family) and the submitted controller
is genuinely online-adaptive - it just fitted the wrong batch-response shape. (An earlier version of this
note read those 3 calls as "scheduling with almost no profiling". That was wrong, and wrong in an
instructive way: the call COUNT is low because profiling is ONE call, not because profiling was skipped.
Reading a lab-call count as an effort level is the same error as reading api_errors as a cutoff.)

The general lesson, and the reason this is written down rather than quietly reverted: **a hold-out rule
must key on whether the agent was CUT OFF, not on whether it was SLOW.** api_errors correlates with how
badly the gateway happened to serve a given model that hour, so holding out on it silently reweights the
comparison towards the models with the better connection - the precise selection effect the `truncated`
field exists to prevent. Terminal failure has its own status (`api_failed`, now in the status tuple);
everything else is a run that continued.

Both runs that prompted the original audit are resolved and stay IN the tally: `k6f2__gpt55__r1` (fail,
R1_time) and `k6w1__sonnet5__r1` (fail, R1_time, and k6w1 is excluded for other reasons).

## Two carried audit items, closed

**`k3v1__gemflash__r1` - 733 lab calls at spent=1.0, graded `truncated: false`.** Correct as graded, and
the asymmetry with the other budget exhaustions is real rather than a bug. Its state is `done` with 92
model calls and an empty final message: the model stopped on its own. The trajectory shows it spent its
last turns fitting scaling laws in `/tmp/loss_interval_analysis.py` and `/tmp/best_fits_detail.py`, got
answers (predictions 3.19-3.71 for the two widths), and never wrote `/app/config.json` - so it fails
`R0_artifact` having done the work. Running out of the LAB budget is the task's own constraint and a
legitimate way to fail; running out of the HARNESS's wall clock is not. Only the latter is `truncated`.

**Four k1 failures are md5-identical unmodified starter stubs** (`eb440111`, 9 lines, the
`judge_win` mean with a +-0.05 interval): `k1c1__gemflash__r1`, `k1c1__sonnet5__r1`, `k1c2__gpt55__r1`,
`k1c2__sonnet5__r1`. All four are already held out - three `timeout`, one `max_calls`. The stub is
therefore evidence about the harness slice, not about the models: none of them declined to edit the file,
they were stopped before they got to it. No further action, but worth stating explicitly, because an
identical-stub cluster looks like a model-behaviour finding and is not one here.

## k8 `partial_fix`: a decoy that never played its strategy (2026-09-28)

Found while collecting gate numbers for the report, by reading the gate LOG instead of its verdict. The
line looked healthy:

    partial_fix  expect=fail pass=0/2 OK  | F s=0.0 R0_report ERR LabError: insufficient budget: this
                                            call costs 4 re-runs, 3 left

`expect=fail`, got 0/2, gate green. But the failure reason is `R0_report` with a budget error: the decoy
identified the cause by brute force - one `canary` per candidate at 4 re-runs each, 12 of its 20 - and
died before writing `report.json`. It never once submitted a partial fix, so for as long as it has been
in the suite it has been asserting nothing at all. The claim it exists to test - *moving the blamed knob
halfway back does not recover 90% of what is reachable* - was untested.

Fixed by giving it the oracle's cheap identification (three `decontaminate` calls, 1 re-run each) so that
the only thing separating it from the oracle is the half-hearted fix. It now fails on `R1_fix_recovers`
at s = 0.61-0.77 across the three shipped instances (19, 27, 78), all of which still ACCEPT.

This is the same shape as `copy_doc` and `read_instruction`, one turn further in: **a gate only tests
what a strategy PLAYS, and "expect=fail, got fail" does not establish that it played.** The verdict column
cannot distinguish "the bad reasoning was caught" from "the decoy crashed"; only the failing-item column
can, and it has to be read. Worth noting the budget error was itself informative - it says identification
by canary alone does not fit in 20 re-runs, which is a property of the task worth knowing and is why the
oracle screens on the cheap offline signal first.

## k1 `wide`: a decoy that duplicated another decoy (2026-09-28)

Found by the same read-the-failing-ITEM pass that caught k8's `partial_fix`. The `wide` strategy is
documented as "right idea, no confidence: report the stratified point but hedge the interval to the cap",
and its purpose is to establish that an interval which costs nothing to state cannot pass. It was failing
on `R1_coverage`, not on `R2_width`.

The reason is that the code did not do what its own comment said. `EST_WIDE` was a hand-written stub that
submitted the **raw judge mean** with a +-WIDTH_CAP/2 interval - so what it actually failed on was the
judge's bias, which is exactly and only what `judge_mean` already tests. Two decoys, one claim, and the
width claim untested.

Rebuilt as `_ppi("prod", "prod", 1, 12.0)`: the ORACLE estimator with the z multiplier inflated ~9x. The
point estimate is now correct by construction, so width is the only thing that can fail, and it does -
`R2_width` alone on both shipped instances (seeds 11 and 14), which still ACCEPT with every other decoy
unchanged. `EST_WIDE` is deleted rather than left unused.

Generalisation, and the reason two of these turned up in one pass: **a decoy's identity is the ITEM it
fails on, not the fact that it fails.** A suite can look complete - one strategy per failure mode, all red -
while two of the strategies are the same experiment and one of the modes has no coverage at all. The
per-item failure column in the gate log is the only place this is visible, and until this session nothing
was reading it. k3, k4 and k6 were audited the same way and are clean: every decoy there fails on its own
substantive item, and only `nop`/`crash` fail on R0, which is correct for them.
