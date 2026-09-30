---
task: t01b-lr-horizon-tight-ws1
blueprint: t01b-lr-horizon-tight
world_seed: 1
review_status: pending
reviewer: 
---

# Review card: Learning-rate transfer to an over-trained production run (tight budget)

**Claim tested.** separate two causes that co-vary in prior runs by designing off-diagonal experiments; state what the lab cannot tell

**Cards.** C1 (Chinchilla-type loss surface), C2 (Learning-rate bowl with scale- and horizon-dependent optimum), C10 (Heteroscedastic seed noise)

**Obstacles.** O1, O9, O5

**Design note.** T01b lr-horizon, tight budget: the T01 world, notebook and questions, but the lab allows 15 runs instead
of 40 (a controlled budget ablation of T01).  A full N x D x lr factorial is no longer affordable: the
notebook's diagonal sweeps have to carry the N-dependence, and the agent's own runs have to be spent
off the diagonal where the horizon effect is visible.  Everything else is identical to T01 (same draw,
same notebook construction, same items, same rivals), so the score difference between T01 and T01b on
matched world seeds is the effect of the budget alone.

## Items, keys, tolerances

| id | kind | key | tol | oracle pass (fresh reps) | rivals killed with margin |
|---|---|---|---|---|---|
| q1 | point | [-3.04, -3.04] | 0.03 | 1.00 | B0_textbook, drop:C2:gD, drop:C2:gN, drop:C1:textbook, widen_points |
| q2 | point | [-3.43, -3.43] | 0.0755 | 1.00 | B0_textbook, B1_notebook_naive, B2_notebook_allcards, B4_more_diagonal, drop:C2:gD, drop:C1:textbook, widen_points |
| q3 | point | [0.05648, 0.05648] | 0.0308 | 1.00 | B2_notebook_allcards, B5_notebook_horizon_only, B4_more_diagonal, widen_points |
| q4 | set | [-3.43, -2.979] | 0.0755 | 1.00 | B0_textbook, B1_notebook_naive, B2_notebook_allcards, B4_more_diagonal, drop:C2:gD, drop:C1:textbook, collapse_sets |
| q5 | verdict | refuted | regret<=0 | 1.00 | B1_notebook_naive |
| q6 | verdict | supported | regret<=0 | 1.00 | B0_textbook, B1_notebook_naive, drop:C2:gD, drop:C1:textbook |
| q7 | decision | C | regret<=0.00217 | 1.00 | B0_textbook, B2_notebook_allcards, B5_notebook_horizon_only, B4_more_diagonal, drop:C2:gD, drop:C1:textbook |
| q8 | point | [2.534, 2.534] | 0.0602 | 1.00 | B0_textbook, B1_notebook_naive, B2_notebook_allcards, drop:C1:textbook, widen_points |

## Rivals

| rival | score | killed with margin on | gated |
|---|---|---|---|
| B0_textbook | 0.12 | q1, q2, q4, q6, q7, q8 | yes |
| B1_notebook_naive | 0.25 | q2, q4, q5, q6, q8 | yes |
| B2_notebook_allcards | 0.38 | q2, q3, q4, q7, q8 | info |
| B5_notebook_horizon_only | 0.38 | q3, q7 | yes |
| B3_random_design | 1.00 | - | info |
| B4_more_diagonal | 0.50 | q2, q3, q4, q7 | info |
| drop:C2:gD | 0.25 | q1, q2, q4, q6, q7 | yes |
| drop:C2:gN | 0.88 | q1 | yes |
| drop:C1:textbook | 0.12 | q1, q2, q4, q6, q7, q8 | yes |
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

