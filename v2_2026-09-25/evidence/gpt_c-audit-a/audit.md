# Audit of the requested contrasts

All estimates use `delta = mean(to) - mean(from)` and hold the other nine launch settings at the query baseline.

## Estimator and validation

I fit validation loss by least squares using categorical launch settings, the ten observed joint categories of the two pre-flight hardware probes, and all pairwise interactions among these factors. `offload` is omitted from the regression because it equals `zero_stage - 1` on every row; the retained coefficient represents their joint movement, not a separately identified effect. Predictions for each query endpoint are averaged over the empirical hardware distribution of all 6,000 jobs, then subtracted. Every reported contrast is orthogonal to the design's null space.

Only pre-run hardware measurements are adjustment variables. Gradient norm, throughput, and step time are measured during training and could mediate effects; adjusting for them would change the requested total-effect estimand. Job identifiers are not predictors. The two hardware generations comprise 3,273 and 2,727 jobs; probe categories within these generations are retained separately.

Exact ten-setting endpoint counts are small, so the regression pools information across observed configurations. This estimates the supported contrasts using measured-hardware adjustment and a pairwise response model; it is not a raw difference of the few exact-matching runs. As with observational adjustment, the causal interpretation relies on the recorded hardware probes adequately accounting for placement differences. Five-fold prediction RMSE is 0.004436 for this model, versus 0.017199 for main effects alone. A compact model retaining only learning-rate/warmup and micro-batch/ZeRO interactions gives RMSE 0.004372 and changes every reported contrast by less than 0.0005. Estimated regression standard errors below quantify sampling variation within the response model.

No estimate is reported for absent requested levels or combinations, even where a restricted parametric model could extrapolate a number.

## Query results

### q01: lr_scale, 1.0 → 2.0

**Identified; delta = -0.02621050.** Hardware-standardized pairwise regression as described above; approximate standard error 0.000451. Exact baseline endpoint counts (from, to) are (2, 2). The learning-rate/warmup interaction is evaluated at warmup 1000. Hardware adjustment matters: the exact baseline matches are on older hardware, while the exact higher-learning-rate matches are on newer hardware.

### q02: warmup, 200 → 1000

**Identified; delta = -0.02952214.** Hardware-standardized pairwise regression as described above; approximate standard error 0.000404. Exact baseline endpoint counts (from, to) are (4, 2). The warmup contrast is evaluated at learning-rate scale 1, allowing warmup effects to differ at other learning rates.

### q03: micro_bs, 8 → 16

**Identified; delta = -0.01525687.** Hardware-standardized pairwise regression as described above; approximate standard error 0.000516. Exact baseline endpoint counts (from, to) are (2, 11). Both micro-batch levels are observed at the requested baseline. The model includes the micro-batch/ZeRO interaction and evaluates it at ZeRO stage 1 and offload 0.

### q04: grad_accum, 1 → 2

**Identified; delta = -0.02219395.** Hardware-standardized pairwise regression as described above; approximate standard error 0.000435. Exact baseline endpoint counts (from, to) are (2, 6). Both accumulation levels occur at the requested baseline. This is a total loss effect, allowing changes in gradient norm and performance during training to contribute.

### q05: zero_stage, 1 → 2

**Underdetermined: `aliased_with_other_knob`.** All 6,000 rows satisfy `offload = zero_stage - 1`: 2,979 rows have (stage 1, offload 0), and 3,021 have (stage 2, offload 1). There is no independent stage variation. The requested stage-2/offload-0 endpoint cannot separate the two knobs.

### q06: offload, 0 → 1

**Underdetermined: `aliased_with_other_knob`.** The same exact alias applies. No rows have stage 1 with offload 1, so changing offload while holding stage 1 is inseparable from the stage effect.

### q07: seq_len, 1024 → 2048

**Identified; delta = -0.03997357.** Hardware-standardized pairwise regression as described above; approximate standard error 0.000482. Exact baseline endpoint counts (from, to) are (20, 2). Both sequence lengths occur with the requested epsilon and other baseline settings.

### q08: act_ckpt, 0 → 1

**Identified; delta = -0.00005789.** Hardware-standardized pairwise regression as described above; approximate standard error 0.000443. Exact baseline endpoint counts (from, to) are (2, 23). Both checkpointing values occur at the requested baseline. The adjusted loss contrast is essentially zero.

### q09: dropout, 0.0 → 0.1

**Identified; delta = 0.01480390.** Hardware-standardized pairwise regression as described above; approximate standard error 0.000396. Exact baseline endpoint counts (from, to) are (2, 8). Both dropout values occur at the requested baseline; the adjusted contrast increases loss.

### q10: opt_eps, 1e-08 → 1e-06

**Underdetermined: `combination_never_observed`.** Epsilon 1e-6 occurs in 1,138 rows, but every one has sequence length 1024. The query requires epsilon 1e-6 with sequence length 2048; there are zero such rows. Both values individually exist, but their required combination does not. An epsilon/sequence-length interaction at that endpoint cannot be learned.

### q11: warmup, 1000 → 2000

**Underdetermined: `level_never_observed`.** The only observed warmup values are 200 (2,387 rows) and 1000 (3,613 rows). Warmup 2000 never appears, so its response cannot be recovered from this log.

### q12: lr_scale, 1.0 → 2.0

**Identified; delta = 0.04391337.** Hardware-standardized pairwise regression as described above; approximate standard error 0.000474. Exact baseline endpoint counts (from, to) are (4, 3). The learning-rate/warmup interaction is evaluated at warmup 200. Both learning-rate levels occur at this baseline; the effect has the opposite sign from q01.

