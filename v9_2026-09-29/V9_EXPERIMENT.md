# v9 trajectory-hardening experiment

This branch adds `l15/tasks/d_shift_repair.py`, a new task family for an AI serving incident. The hidden
mechanism is a cohort-mixture shift: an aggregate dashboard can remain flat while a frozen target cohort
regresses. Three release settings are visibly changed and coupled through a calibration optimum. The agent
must use the frozen cohort to identify the causal setting, then submit a joint repair respecting per-setting
caps. The grader separates repair quality from counterfactual attribution.

The mechanism is orthogonal to `k8_post`: k8 uses offline leakage and model knob intervention; this family
uses population measurement validity plus coupled repair. It supports QA (the report), free experiment design
(choosing probes/cohorts), and a future code-artifact variant with the same mechanism card.

## Local evidence

Using the bundled Python runtime with NumPy:

```text
python3 -m l15.v8gates d_shift_repair 0 100 9 v9_reports/d_shift_repair.json
screened 96/100, strategy-gate clean 95
OK
```

The certificate reports a 9-instance admitted pool with router/calibration/features counts 2/3/4, all
strategy, ablation, search, mutation and activity checks passing. The oracle passes 4/4 salts on accepted
instances. The one rejected candidate (seed 95) failed the oracle solvability screen and is excluded.

This is a correctness certificate, not a frontier-model difficulty certificate. GPT-6/Fable runs still need
to be executed through the harness, and failures need trajectory review to confirm they are caused by cohort
measurement and joint-repair reasoning rather than format or transport issues.
