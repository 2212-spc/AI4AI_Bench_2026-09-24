---
task: t03b-filter-repeat-tight-ws11
blueprint: t03b-filter-repeat-tight
world_seed: 11
review_status: pending
reviewer: 
---

# Review card: Quality filtering for a data-constrained production run (tight budget)

**Claim tested.** decompose two effects that cancel in the team's ablation by designing experiments that isolate each; carry a documented range through a nonlinear model

**Cards.** C1 (Chinchilla-type loss surface), C6 (Diminishing value of repeated tokens), C7 (Quality filtering multiplies token value but shrinks the pool), C10 (Heteroscedastic seed noise)

**Obstacles.** O2, O3, O5

**Design note.** T03b filter-repeat, tight budget: the T03 world, notebook and questions with 12 lab runs instead of 40
(a controlled budget ablation of T03).  Separating the quality and repetition constants now has to reuse
the notebook: its masked ablation is one equation in (mu, nu, Rs); a short fresh-data filter sweep and a
two-point repetition sweep supply the others.  Same draw, notebook, items and rivals as T03.

## Items, keys, tolerances

| id | kind | key | tol | oracle pass (fresh reps) | rivals killed with margin |
|---|---|---|---|---|---|
| q1 | point | [0.02177, 0.02177] | 0.00565 | 1.00 | B0_textbook, B1_notebook_naive, B2_notebook_allcards, B5_literature_Rs, B4_replicate_ablation, drop:C6:Rs, drop:C1:textbook, widen_points |
| q2 | point | [0.008754, 0.008754] | 0.00297 | 0.90 | B0_textbook, B1_notebook_naive, B2_notebook_allcards, drop:C6:Rs, drop:C7:mu, drop:C1:textbook, widen_points |
| q3 | decision | A | regret<=0.00183 | 1.00 | B0_textbook, B2_notebook_allcards, drop:C6:Rs |
| q4 | verdict | refuted | regret<=0 | 1.00 | B4_replicate_ablation, drop:C6:Rs |
| q5 | set | [-0.008723, 0.03131] | 0.00851 | 0.90 | B0_textbook, B1_notebook_naive, B2_notebook_allcards, B4_replicate_ablation, drop:C6:Rs, drop:C1:textbook, collapse_sets |
| q6 | verdict | undetermined | regret<=0 | 1.00 | B1_notebook_naive, drop:C6:Rs, drop:C7:mu, drop:C1:textbook |

## Rivals

| rival | score | killed with margin on | gated |
|---|---|---|---|
| B0_textbook | 0.33 | q1, q2, q3, q5 | yes |
| B1_notebook_naive | 0.33 | q1, q2, q5, q6 | yes |
| B2_notebook_allcards | 0.33 | q1, q2, q3, q5 | info |
| B5_literature_Rs | 0.50 | q1 | info |
| B3_random_design | 1.00 | - | info |
| B4_replicate_ablation | 0.33 | q1, q4, q5 | info |
| drop:C6:Rs | 0.00 | q1, q2, q3, q4, q5, q6 | yes |
| drop:C7:mu | 0.33 | q2, q6 | yes |
| drop:C1:textbook | 0.33 | q1, q2, q5, q6 | yes |
| collapse_sets | 0.83 | q5 | yes |
| widen_points | 0.67 | q1, q2 | yes |

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

