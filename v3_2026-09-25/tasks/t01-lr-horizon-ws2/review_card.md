---
task: t01-lr-horizon-ws2
blueprint: t01-lr-horizon
world_seed: 2
review_status: pending
reviewer: 
---

# Review card: Learning-rate transfer to an over-trained production run

**Claim tested.** separate two causes that co-vary in prior runs by designing off-diagonal experiments; state what the lab cannot tell

**Cards.** C1 (Chinchilla-type loss surface), C2 (Learning-rate bowl with scale- and horizon-dependent optimum), C10 (Heteroscedastic seed noise)

**Obstacles.** O1, O9, O5

**Design note.** T01 lr-horizon: a learning-rate rule fitted along the Chinchilla diagonal is applied off-diagonal.

Notebook: LR sweeps at five model sizes, every run at D = 20N (O1: model size and token horizon co-vary).
The lab notes attribute the whole drift of the optimum to model size (a 'muP width bug') and plan a
heavily over-trained production run (1B params, 1T tokens, 4M-token batches) with that rule (O9).
Truth: the optimum depends on both N (weakly) and D (strongly).  Separating them needs off-diagonal
sweeps.  The batch-size dependence of the optimum has zero footprint in the lab (batch is pinned at
0.5M) and is documented only as a range (O5).

## Items, keys, tolerances

| id | kind | key | tol | oracle pass (fresh reps) | rivals killed with margin |
|---|---|---|---|---|---|
| q1 | point | [-2.693, -2.693] | 0.03 | 1.00 | drop:C2:gD, drop:C2:gN, drop:C1:textbook, widen_points |
| q2 | point | [-3.011, -3.011] | 0.0514 | 1.00 | B1_notebook_naive, B5_notebook_horizon_only, B4_more_diagonal, drop:C2:gD, drop:C2:gN, drop:C1:textbook, widen_points |
| q3 | point | [0.01912, 0.01912] | 0.0139 | 1.00 | B5_notebook_horizon_only, B4_more_diagonal, drop:C1:textbook, widen_points |
| q4 | set | [-3.011, -2.56] | 0.0514 | 1.00 | B1_notebook_naive, B5_notebook_horizon_only, B4_more_diagonal, drop:C2:gD, drop:C2:gN, drop:C1:textbook, collapse_sets |
| q5 | verdict | refuted | regret<=0 | 1.00 | B1_notebook_naive, B2_notebook_allcards |
| q6 | verdict | undetermined | regret<=0 | 1.00 | B1_notebook_naive, B5_notebook_horizon_only, B4_more_diagonal, drop:C2:gD, drop:C1:textbook |
| q7 | decision | B | regret<=0.00921 | 1.00 | B1_notebook_naive, B4_more_diagonal, drop:C2:gD, drop:C2:gN, drop:C1:textbook |
| q8 | point | [2.415, 2.415] | 0.0407 | 1.00 | B0_textbook, B4_more_diagonal, drop:C1:textbook, widen_points |

## Rivals

| rival | score | killed with margin on | gated |
|---|---|---|---|
| B0_textbook | 0.88 | q8 | yes |
| B1_notebook_naive | 0.38 | q2, q4, q5, q6, q7 | yes |
| B2_notebook_allcards | 0.50 | q5 | info |
| B5_notebook_horizon_only | 0.38 | q2, q3, q4, q6 | yes |
| B3_random_design | 1.00 | - | info |
| B4_more_diagonal | 0.25 | q2, q3, q4, q6, q7, q8 | info |
| drop:C2:gD | 0.25 | q1, q2, q4, q6, q7 | yes |
| drop:C2:gN | 0.38 | q1, q2, q4, q7 | yes |
| drop:C1:textbook | 0.12 | q1, q2, q3, q4, q6, q7, q8 | yes |
| collapse_sets | 0.88 | q4 | yes |
| widen_points | 0.50 | q1, q2, q3, q8 | yes |

## Gates

- G1_solvable: **pass**
- G2_kill_matrix: **pass**
- G3_set_width: **pass**
- G4_identifiability: **pass**
- G5_replay: **pass**
- G6_leakage: **pass**
- G7_key_types: **pass**
- G8_decision_gap: **pass**
- G9_load_bearing: **pass**
- G10_item_useful: **pass**

## Human checklist (tick each; any 'no' -> reject or fix the blueprint)

- [ ] The notebook and lab notes read like a plausible team artefact; nothing in them states the answer.
- [ ] Each question is unambiguous given manual.md Section 3 (point vs. set vs. verdict semantics).
- [ ] Each known unknown is truly undeterminable in this lab (see cert.json G4) and its documented range is the one used for the key.
- [ ] The keys follow from the cards' stated forms (spot-check one numeric key by hand from hidden/world.json).
- [ ] The oracle's runs (solution/oracle_runs.jsonl) are a design a competent researcher could think of.
- [ ] The rival that encodes the notes' reading fails for the reason the design note says.
- [ ] Grounding references for each active card are appropriate (cards.py).

Reviewer notes:

