# a1-da4b92 (family A1: masked convention-conflict regression)

Not shown to the agent.

## Difficulty
Two convention bugs in the v2 trainer partially compensate each other (masking): fixing only one makes results worse or
diverge, and a learning-rate rescale can match the canonical number while failing held-out configurations.
Empirically this family is saturated by frontier agents (4/4 passes) and is kept as a calibrated negative control.

## Reference solution
`solution/fix.patch` restores the reference semantics described in the README (applied by solve.sh).

## Verification
The verifier imports the agent's `minilab.trainer.train` (the artifact), trains on the canonical and held-out
configurations with fresh data from the teacher and compares mean held-out MSE with reference statistics computed by
the generator's reference implementation (`authoring/certificate.json` lists the near-miss variants that must fail:
single fixes, lr-compensation hacks, decoy edits).
