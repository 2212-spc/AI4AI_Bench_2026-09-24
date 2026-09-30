---
task: t03-filter-repeat-ws28
blueprint: t03-filter-repeat
world_seed: 28
review_status: pending
reviewer: 
---

# Review card: Quality filtering for a data-constrained production run

**Claim tested.** decompose two effects that cancel in the team's ablation by designing experiments that isolate each; carry a documented range through a nonlinear model

**Cards.** C1 (Chinchilla-type loss surface), C6 (Diminishing value of repeated tokens), C7 (Quality filtering multiplies token value but shrinks the pool), C10 (Heteroscedastic seed noise)

**Obstacles.** O2, O3, O5

**Design note.** T03 filter-repeat: two effects cancel in the team's ablation (masking), then diverge at production.

Notebook: the team's quality-filter ablation ran on a small raw subsample for several epochs.  Filtering
raises the value of each token (C7) but shrinks the unique pool, so the ablation repeats data more (C6).
At the ablation's epoch count the two effects cancel by construction (O2): the notes conclude 'filtering
does nothing' and plan a 5-10 epoch production run without it.  At production the balance is different
(the regime is reachable in the lab only if one emulates scarcity with a small `sub`; O3).  Separating
the two effects needs a fresh-data filter sweep and a repetition sweep.  The next corpus snapshot's
size is documented only as a range (O5): production quantities that depend on it are sets.

## Items, keys, tolerances

| id | kind | key | tol | oracle pass (fresh reps) | rivals killed with margin |
|---|---|---|---|---|---|
| q1 | point | [0.01335, 0.01335] | 0.00166 | 1.00 | B0_textbook, B1_notebook_naive, B2_notebook_allcards, B5_literature_Rs, drop:C6:Rs, drop:C1:textbook, widen_points |
| q2 | point | [-0.006738, -0.006738] | 0.0032 | 1.00 | B1_notebook_naive, drop:C6:Rs, drop:C7:mu, drop:C1:textbook, widen_points |
| q3 | decision | B | regret<=0.00199 | 1.00 | B1_notebook_naive, drop:C6:Rs, drop:C7:mu, drop:C1:textbook |
| q4 | verdict | supported | regret<=0 | 1.00 | B1_notebook_naive, drop:C7:mu, drop:C1:textbook |
| q5 | set | [-0.01502, 0.003009] | 0.00418 | 1.00 | B1_notebook_naive, drop:C6:Rs, drop:C7:mu, drop:C1:textbook, collapse_sets |
| q6 | verdict | refuted | regret<=0 | 0.90 | B0_textbook, B2_notebook_allcards, B5_literature_Rs, drop:C6:Rs |

## Rivals

| rival | score | killed with margin on | gated |
|---|---|---|---|
| B0_textbook | 0.50 | q1, q6 | yes |
| B1_notebook_naive | 0.17 | q1, q2, q3, q4, q5 | yes |
| B2_notebook_allcards | 0.33 | q1, q6 | info |
| B5_literature_Rs | 0.33 | q1, q6 | info |
| B3_random_design | 1.00 | - | info |
| B4_replicate_ablation | 1.00 | - | info |
| drop:C6:Rs | 0.17 | q1, q2, q3, q5, q6 | yes |
| drop:C7:mu | 0.33 | q2, q3, q4, q5 | yes |
| drop:C1:textbook | 0.17 | q1, q2, q3, q4, q5 | yes |
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

