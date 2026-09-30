# Quality/repetition analysis

Read the manual and notebook before experiments. Used 40 runs costing
7.8912e19 FLOPs, within the 8e19 budget. Raw observations are in
`lab_runs.jsonl`; experiment plans are `plan1.json` through `plan3.json`.

## Experimental controls

- Fresh q=0 runs independently varied N and D to identify data scaling.
- Fresh-data q sweeps isolated quality from unique-pool shrinkage.
- q=0 subset sweeps measured repetition over 2--50 epochs.
- Joint q/sub comparisons reproduced five raw epochs, the specified production
  snapshot's ratio, and two and ten raw epochs, the next snapshot endpoints.
- Higher-N runs reduced noise and checked that the quality/repetition findings
  persist across model sizes.

The notebook ablations mix quality gains and repetition penalties. Their nearly
zero net difference does not establish that filtering has no intrinsic value.
The claimed repetition rule from literature is not a lab guarantee.

## Fitted relationships and cross-checks

A separable loss model fits all notebook and new measurements:

    loss = floor + A*(N/1e8)^(-alpha) + B*(effective_D/1e10)^(-beta)
    U = sub*(1-q)
    effective_D = [min(D,U) + U*R*(1-exp(-max(D/U-1,0)/R))]*(1+k*q^2)

Fitted parameters are approximately floor=2.019, A=1.166, alpha=0.334,
B=0.740, beta=0.302, R=6.315, k=1.029. Residuals are consistent with observed
seed noise. An alternative logarithmic repetition model fits appreciably worse.
Allowing the quality exponent to vary gives 2.116 and leaves the answer estimates
nearly unchanged. `analyze.py` contains the numerical fit and checks.

At the specified production snapshot, fitted repetition cost is 0.0218 nats.
The q=0.5 effect is about +0.008--0.009 nats. Relative effects for the decision
options q=0,0.3,0.6,0.85 are approximately 0,+0.0049,+0.0138,+0.0797 nats.
The fresh-data q=0.6 effect at D=1e11 is about -0.0335 nats, less than the
0.050-nat improvement required by q4.

For q5, evaluating the fitted effect across the entire documented snapshot range
finds its minimum at sub=1e11 (about -0.0091 nats) and maximum at sub=2e10
(about +0.0306 nats). These are range endpoints, not statistical bounds.
For q6, the corresponding q=0.6 effects have opposite signs: about -0.0109 nats
at sub=1e11 and +0.0439 nats at sub=2e10. The unknown snapshot size therefore
makes q6 undetermined. Fixed-snapshot numerical answers are degenerate intervals,
as required by the manual; they are not widened for seed noise.
