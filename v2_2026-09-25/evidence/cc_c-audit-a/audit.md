# Audit: what `log.csv` can and cannot answer

## Method (shared by every identified query)

* Adjustment set: the 9 free launch knobs plus the two pre-flight hardware probes
  (`nccl_bw_gbps`, `sm_clock_mhz`). The probes are measured before any setting takes effect and
  are the only record of placement, which was non-random: `micro_bs`, `seq_len`, `act_ckpt` and
  `opt_eps` are all strongly tilted toward the newer hardware generation, and the newer generation
  has val_loss about 0.046 lower. So hardware is a confounder and must be conditioned on.
* Excluded: `grad_norm_p95`, `throughput_toks_s`, `step_time_ms` (measured during the run, so they
  are downstream of the knobs; conditioning on them would block part of the effect).
* `offload` is dropped from the regressors because it is a copy of `zero_stage` (see q05/q06).
* Model: OLS on the outcome with additive dummies for every knob level and hardware level, plus the
  interactions that a full pairwise screen flagged: `lr_scale x warmup` (huge, F ~ 19000) and
  `micro_bs x zero_stage` (F ~ 275). After adding those two, residual sd drops from 0.0172 to
  0.0044, no further pairwise or three-way term is significant, residual means are zero within
  every knob level and every hardware cell, and residuals are uncorrelated with the mediators.
  The hardware x zero_stage terms that appeared in the first screen vanish once the lr x warmup
  term is in the model.
* Estimator: for each query, predict val_loss at the baseline settings with knob=`to` and with
  knob=`from`, for every one of the 6000 observed hardware profiles, and average the difference.
  This averages the contrast over the cluster's own hardware mix, as requested. Because no knob
  interacts with hardware, the average equals the single coefficient contrast.
* Robustness: refitting with all 36 knob-knob pairwise interactions, or all 55 knob+hardware
  pairwise interactions, moves every reported delta by < 0.0006. Coefficient standard errors are
  ~0.0002-0.0003.

## Per-query

* **q01 lr_scale 1.0 -> 2.0 @ warmup=1000**: identified, delta = -0.0258.
  lr x warmup interaction is decisive; at warmup=1000 the contrast is (-0.0061 - 0.0700) - (-0.0505).
* **q02 warmup 200 -> 1000 @ lr_scale=1.0**: identified, delta = -0.0300.
  At lr_scale=1.0 the interaction term is ~0, so this is the warmup main effect.
* **q03 micro_bs 8 -> 16 @ zero_stage=1**: identified, delta = -0.0155.
  Strongly confounded by placement (micro_bs=16 jobs land on fast nodes; raw gap is -0.037);
  hardware adjustment removes it. micro_bs x zero_stage interaction is evaluated at zero_stage=1.
* **q04 grad_accum 1 -> 2**: identified, delta = -0.0219. Additive main effect, no interactions.
* **q05 zero_stage 1 -> 2**: underdetermined, `aliased_with_other_knob`.
  In all 6000 rows `offload == zero_stage - 1`; the pairs (1,1) and (2,0) never occur. Whatever
  is attributed to zero_stage could equally be offload; no sample size fixes this.
* **q06 offload 0 -> 1**: underdetermined, `aliased_with_other_knob`. Same aliasing as q05.
* **q07 seq_len 1024 -> 2048 @ opt_eps=1e-08**: identified, delta = -0.0400.
  Both seq_len levels are observed at opt_eps=1e-08 (2107 and 2755 runs); refitting on only those
  rows gives the same -0.0400. Placement confounding (seq_len=2048 favours fast nodes) is adjusted.
* **q08 act_ckpt 0 -> 1**: identified, delta = -0.0001 (indistinguishable from zero).
  The raw gap of -0.053 is entirely a placement artefact: act_ckpt=1 jobs are on newer hardware.
* **q09 dropout 0.0 -> 0.1**: identified, delta = +0.0150. Additive main effect.
* **q10 opt_eps 1e-08 -> 1e-06 @ seq_len=2048**: underdetermined, `combination_never_observed`.
  opt_eps=1e-06 appears only with seq_len=1024 (1138 runs); (seq_len=2048, opt_eps=1e-06) has
  zero rows. The requested contrast needs that cell, so it rests on an untestable assumption that
  the opt_eps effect carries over unchanged to seq_len=2048.
* **q11 warmup 1000 -> 2000**: underdetermined, `level_never_observed`. warmup takes only
  {200, 1000} in the log.
* **q12 lr_scale 1.0 -> 2.0 @ warmup=200**: identified, delta = +0.0444.
  Same knob as q01 but at warmup=200 the interaction is absent, so doubling the lr hurts:
  (-0.0061) - (-0.0505). The sign flip versus q01 is the lr x warmup interaction.
