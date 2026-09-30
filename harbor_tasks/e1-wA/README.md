# e1-wA (family E1: budgeted recipe-change triage with interacting changes)

Not shown to the agent.

## Difficulty
Six candidate recipe changes for a small MLP trainer interact: some pairs diverge together, some changes only help
in combination, one change (c6) is a pure RNG-stream decoy. The budget (80 lab runs) is far below what
naive per-arm replication needs, the teammate note is an honest single-seed one-factor-at-a-time ablation whose plan
is wrong, and the textbook path (single ablations, then fix instabilities one knob at a time) ends in a local optimum.
The optimum is c1+c2+c3+c5 (mean val MSE 0.04200); accept set within 4.5%:
c1+c2+c3, c1+c2+c3+c5, c1+c2+c3+c6, c1+c2+c3+c5+c6.

Simulated policies against the truth table (pass rate, typical end states):

| policy | pass | typical end |
|---|---|---|
| expert_full_factorial | 0.998 | c1+c2+c3+c5 (0.51), c1+c2+c3+c5+c6 (0.47) |
| expert_prune_c6 | 1.000 | c1+c2+c3+c5 (1.00), c1+c2+c3 (0.00) |
| ofat_combine | 0.000 | c1+c2+c4+c5 (0.57), c1+c2+c4+c5+c6 (0.43) |
| ofat_then_verify | 0.000 | c1 (0.93), c2 (0.07) |
| greedy_forward | 0.007 | c1+c5 (0.45), c1+c5+c6 (0.35) |
| greedy_backward | 1.000 | c1+c2+c3+c5 (0.55), c1+c2+c3+c5+c6 (0.45) |
| ship_all | 0.000 | c1+c2+c3+c4+c5+c6 (1.00) |
| ship_none | 0.000 | (none) (1.00) |
| random_subset | 0.073 | c3+c4+c5 (0.03), c1+c5 (0.03) |

## Reference solution
`solution/expert_e1.py`: screens all 32 subsets of c1-c5 on common seeds (c6 dropped by reasoning: it only changes
the minibatch RNG stream), replicates the leaders on fresh common seeds with the remaining budget, ships the best
and predicts its mean. It uses only public information and the lab CLI. Certified pass rate
1.000 (simulation over the truth table).

## Verification
`tests/verify_e1.py` reads only `/app/decision.json` (artifact) and the baked-in truth `tests/scoring.json`
(64 subsets x 80 seeds, common random numbers, computed by executing the trainer). Reward 1 iff the
shipped subset's regret <= 4.5% and the predicted val MSE is within 6% of its true mean.
The accept boundary sits in a gap of the sorted regrets (2.7 paired SE clearance).
The budget is enforced by the lab sidecar (HTTP 429), whose ledger is not visible to the agent.
Evidence: `authoring/certificate.json` (C1 truth resolution, C2 structure, C3 blind solvability, C4 no shortcut, C5 estimation).
