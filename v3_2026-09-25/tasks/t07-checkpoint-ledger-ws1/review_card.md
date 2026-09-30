---
task: t07-checkpoint-ledger-ws1
blueprint: t07-checkpoint-ledger
world_seed: 1
review_status: pending
reviewer: 
---

# Review card: Scaling law from intermediate checkpoints, carried to a production run that may be stopped early

**Claim tested.** recognise that intermediate checkpoints and finished runs are different measurements, infer how the gap depends on the schedule, design unbiased measurements within a tight budget, and carry both the scaling law and the schedule effect to a production run with an unknown stopping point

**Cards.** C1 (Chinchilla-type loss surface), C11 (Schedule-dependent intermediate-checkpoint loss), C10 (Heteroscedastic seed noise)

**Obstacles.** O7, O1, O5, O9

**Design note.** T07 checkpoint-ledger: a scaling law built from intermediate checkpoints (O7), carried to production.

Notebook: the team trained one long cosine run per size and logged intermediate checkpoints.  They fitted
L(N, D) treating every checkpoint as if it were a finished run of f*D tokens (the Kaplan-style shortcut).
Under C11 a checkpoint at fraction f carries a schedule penalty ca*r(f)^zeta that shrinks as the learning
rate decays, so early checkpoints look worse than finished short runs and the fitted data exponent is
biased; the team's compute-optimal allocation is wrong (O7).  Even finished cosine runs carry a residual
penalty ca*rmin^zeta, which a fit on cosine finals silently absorbs into E; the production question is
about a WSD run (no residual), so the offset must be measured (WSD cooldown branches give unbiased
finished-run losses cheaply; stable-phase WSD checkpoints give ca directly).  The team's WSD pilot looks
worse than cosine at every mid-run checkpoint (the stable phase carries the full penalty) although its
finished loss is lower - the notes misread this (O1).  Production may be stopped early at an unknown
fraction of its steps (known unknown, O5): the loss at the stop is a set.

## Items, keys, tolerances

| id | kind | key | tol | oracle pass (fresh reps) | rivals killed with margin |
|---|---|---|---|---|---|
| q1 | decision | D | regret<=0.00561 | 1.00 | B1_notes_reading |
| q2 | point | [1.966, 1.966] | 0.029 | 1.00 | B0_textbook, drop:C1:textbook, widen_points |
| q3 | point | [0.08518, 0.08518] | 0.00379 | 1.00 | B0_textbook, B1_notes_reading, B2_cosine_finals_no_schedule, B3_linear_in_lr, drop:C11:ca, drop:C1:textbook, widen_points |
| q4 | set | [0.01149, 0.1125] | 0.003 | 1.00 | B0_textbook, B1_notes_reading, B2_cosine_finals_no_schedule, drop:C11:ca, drop:C1:textbook, collapse_sets |
| q5 | verdict | refuted | regret<=0 | 1.00 | B1_notes_reading, B2_cosine_finals_no_schedule, drop:C11:ca, drop:C1:textbook |

## Rivals

| rival | score | killed with margin on | gated |
|---|---|---|---|
| B0_textbook | 0.40 | q2, q3, q4 | yes |
| B1_notes_reading | 0.20 | q1, q3, q4, q5 | yes |
| B2_cosine_finals_no_schedule | 0.40 | q3, q4, q5 | yes |
| B3_linear_in_lr | 0.80 | q3 | info |
| B4_cosine_grid_full_law | 1.00 | - | info |
| drop:C11:ca | 0.20 | q3, q4, q5 | yes |
| drop:C1:textbook | 0.20 | q2, q3, q4, q5 | yes |
| collapse_sets | 0.80 | q4 | yes |
| widen_points | 0.60 | q2, q3 | yes |

## Gates

- G0_wellposed: **pass**
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

