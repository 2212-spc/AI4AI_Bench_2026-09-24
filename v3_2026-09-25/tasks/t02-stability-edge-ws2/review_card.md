---
task: t02-stability-edge-ws2
blueprint: t02-stability-edge
world_seed: 2
review_status: pending
reviewer: 
---

# Review card: Stability of a 7B production run planned from small-scale sweeps

**Claim tested.** bound a quantity the lab only observes censored (a stability edge that never binds in the lab), choose the experiment that makes the bound tightest, and state exactly what remains unknown

**Cards.** C1 (Chinchilla-type loss surface), C2 (Learning-rate bowl with scale- and horizon-dependent optimum), C4 (Divergence edge (attention-logit / output-logit instability)), C10 (Heteroscedastic seed noise)

**Obstacles.** O3, O5, O9

**Design note.** T02 stability-edge: a learning rate that is safe in every lab sweep diverges at production scale, and
the fix the team dismissed (qk-layernorm) can only be bounded, not measured, in this lab.

Notebook: LR sweeps along D = 20 N at four sizes (warmup 2%, qk-layernorm off) plus a paired qk on/off
check at one size.  The largest-LR points at the biggest sizes diverged; the notes call that the usual
instability far above the optimum.  qk-layernorm made no difference to loss, so the notes drop it.  They
extrapolate the loss-optimal LR to a 7B run and plan to use it.

Truth: the divergence edge falls with N faster than the optimal LR does (O3/O9): at 7B the edge is
below the optimum, so the loss-optimal LR is not runnable without qk-layernorm.  With qk-layernorm on,
no lab run can diverge (the lab's LR cap sits below the qk-on edge everywhere), so the qk factor Q is
*censored*: the lab gives a lower bound Q >= lr_cap / edge_off(most unstable lab config), the stack's
documentation gives the upper bound (O5).  The tightest lower bound needs the qk-on run at the lab
corner that is least stable without qk (largest N, zero warmup, LR cap) - a design choice.  The edge is
deterministic in this lab (div_jitter = 0) so the identification set has crisp endpoints.

## Items, keys, tolerances

| id | kind | key | tol | oracle pass (fresh reps) | rivals killed with margin |
|---|---|---|---|---|---|
| q1 | point | [-3.136, -3.136] | 0.025 | 1.00 | B0_textbook, B1_notes_reading, drop:C4:delta, widen_points |
| q2 | set | [-1.839, -1.358] | 0.025 | 1.00 | B0_textbook, B1_notes_reading, B2_doc_range_only, B3_bound_at_prod_warmup, B4_bound_at_mid_size, B5_bound_as_point, B6_probe_below_cap, B7_no_warmup_effect, drop:C4:delta, drop:C4:warmup, collapse_sets |
| q3 | verdict | supported | regret<=0 | 1.00 | B0_textbook, B1_notes_reading, drop:C4:delta |
| q4 | verdict | supported | regret<=0 | 1.00 | B1_notes_reading, B2_doc_range_only |
| q5 | verdict | undetermined | regret<=0 | 1.00 | B1_notes_reading, B5_bound_as_point, drop:C4:delta |
| q6 | decision | C | regret<=0.00575 | 1.00 | B0_textbook, B1_notes_reading, drop:C4:delta, drop:C2:bowl |
| q7 | verdict | supported | regret<=0 | 1.00 | B7_no_warmup_effect, drop:C4:warmup |

## Rivals

| rival | score | killed with margin on | gated |
|---|---|---|---|
| B0_textbook | 0.43 | q1, q2, q3, q6 | yes |
| B1_notes_reading | 0.14 | q1, q2, q3, q4, q5, q6 | yes |
| B2_doc_range_only | 0.71 | q2, q4 | yes |
| B3_bound_at_prod_warmup | 0.86 | q2 | yes |
| B4_bound_at_mid_size | 0.86 | q2 | info |
| B5_bound_as_point | 0.71 | q2, q5 | yes |
| B6_probe_below_cap | 0.86 | q2 | info |
| B7_no_warmup_effect | 0.71 | q2, q7 | yes |
| drop:C4:delta | 0.29 | q1, q2, q3, q5, q6 | yes |
| drop:C4:warmup | 0.71 | q2, q7 | yes |
| drop:C2:bowl | 0.86 | q6 | yes |
| collapse_sets | 0.86 | q2 | yes |
| widen_points | 0.86 | q1 | yes |

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

