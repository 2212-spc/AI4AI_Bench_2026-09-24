# e1-w14-b40 (family E1: budgeted recipe-change triage with interacting changes)

Not shown to the agent.

## Difficulty
Six candidate recipe changes for a small MLP trainer interact: some pairs diverge together, some changes only help
in combination, one change (c6) is a pure RNG-stream decoy. The budget (40 lab runs) is far below what
naive per-arm replication needs, the teammate note is an honest single-seed one-factor-at-a-time ablation whose plan
is wrong, and the textbook path (single ablations, then fix instabilities one knob at a time) ends in a local optimum.
The optimum is c1+c2+c3+c5+c6 (mean val MSE 0.03995); accept set within 3.1%:
c1+c2+c3+c5, c1+c2+c3+c5+c6.

Simulated policies against the truth table (pass rate, typical end states):

| policy | pass | typical end |
|---|---|---|
| expert_full_factorial | 0.000 | (none) (1.00) |
| expert_prune_c6 | 0.985 | c1+c2+c3+c5 (0.99), c1+c2+c3+c4+c5 (0.01) |
| ofat_combine | 0.000 | c5+c6 (0.50), c5 (0.45) |
| ofat_then_verify | 0.000 | c5 (0.52), c5+c6 (0.45) |
| greedy_forward | 0.138 | c1+c5 (0.29), c5 (0.18) |
| greedy_backward | 0.997 | c1+c2+c3+c5+c6 (0.59), c1+c2+c3+c5 (0.41) |
| ship_all | 0.000 | c1+c2+c3+c4+c5+c6 (1.00) |
| ship_none | 0.000 | (none) (1.00) |
| random_subset | 0.027 | c4+c5 (0.03), c2+c3+c4+c5 (0.03) |

## Reference solution
`solution/expert_e1.py`: screens all 32 subsets of c1-c5 on common seeds (c6 dropped by reasoning: it only changes
the minibatch RNG stream), replicates the leaders on fresh common seeds with the remaining budget, ships the best
and predicts its mean. It uses only public information and the lab CLI. Certified pass rate
0.985 (simulation over the truth table).

## Verification
`tests/verify_e1.py` reads only `/app/decision.json` (artifact) and the baked-in truth `tests/scoring.json`
(64 subsets x 80 seeds, common random numbers, computed by executing the trainer). Reward 1 iff the
shipped subset's regret <= 3.1% and the predicted val MSE is within 6% of its true mean.
The accept boundary sits in a gap of the sorted regrets (3.5 paired SE clearance).
The budget is enforced by the lab sidecar (HTTP 429), whose ledger is not visible to the agent.
Evidence: `authoring/certificate.json` (C1 truth resolution, C2 structure, C3 blind solvability, C4 no shortcut, C5 estimation).
