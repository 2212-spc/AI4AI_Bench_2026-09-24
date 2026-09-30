---
task: e03-audit-headline-ws3
blueprint: e03-audit-headline
world_seed: 3
review_status: pending
reviewer: 
---

# Review card: Audit the release headline

**Claim tested.** The number in the review memo is what the team's own evidence implies, and the release bar is cleared.

**Lab.** evallab

**Cards.** E1 (Two-parameter item response), E2 (Format sensitivity and answer-extraction failure), E3 (Contamination through duplicated items), E6 (Between-call variance and the resulting detectable effect size), E7 (Slice mixture and aggregation convention (Simpson)), E8 (n-gram contamination detector with calibration error)

**Obstacles.** O14, O15, O4, O16, O10, O5

**Design note.** Audit a teammate's analysis script: one silent defect, three provably harmless red herrings.

The team has already spent its measurement budget.  What is shipped is the evidence they collected (an
exact corpus scan, a cheap detector table, repeated per-item scoring passes for two models) plus the
script that turned it into the number in the review memo.  Exactly one step of that script makes a wrong
inference.  It runs, it prints a plausible number, no test fails and no log line is anomalous.

Why the keys are exact.  Every question here is a question about *this evidence*: what the shipped rows
imply once the defective step is fixed.  So the key is a deterministic function of the shipped bytes, not
an estimate of the world, and the oracle reproduces it with zero error.  The difficulty is entirely
inferential - find the step, redo the aggregation, and carry a known unknown through to a release
decision - which is the point of the form: a defect that changes no test outcome is invisible to a
test-oracle benchmark, and nothing here crashes.

Which defect is injected is drawn per instance from three of the taxonomy's entries, each realised as a
one-line change in the same script.  The other two ways of being wrong are present in the *world* (a
cheap detector that misses most contamination, an extractor that drops a fifth of one model's answers),
so a naive analyst has three separate chances to produce the wrong number and the injected defect is not
guessable from which mechanisms are active.

Exactly-one-defect check (the audit answer is a closed-set choice, so ambiguity is unfairness).  Against
the nine taxonomy entries, as shipped: D1/D2/D7 are the injected variants, one per instance, and the
other two of the three are correct in that instance.  D3 cannot be injected here (this bank's slices are
drawn uniformly, so micro and macro differ by about 0.002, well under the accepted width) and the script
uses the item-weighted form.  D4 needs a judge; there is none.  D5 needs a reported standard error or
interval; the script reports none.  D6 needs unequal denominators across the averaged units; every
repetition presents the same 1500 items, so mean-of-ratios and ratio-of-means agree to 1e-5.  D8 needs a
threshold-crossing claim; the script makes none (the D7 variant thresholds a noisy score to *filter*,
and the exact scan it ignores is loaded and printed two lines above, which is D7's own text).  D9 needs
failed runs; this lab has none.

## Items, keys, tolerances

| id | kind | key | tol | oracle pass (fresh reps) | rivals killed with margin |
|---|---|---|---|---|---|
| a1 | audit | {"defect": "D1_selection_max_over_seeds", "lo": 0.4386978417266187, "hi": 0.4416978417266187} | regret<=0 | 1.00 | skip:defect, skip:recompute, B_prior, naive_ignore:E2, naive_ignore:E3, naive_ignore:E6, naive_ignore:E8, drop:E2, drop:E7 |
| a2 | set | [0.2439, 0.497] | 0.002 | 1.00 | skip:defect, skip:recompute, skip:weights, B_prior, naive_ignore:E2, naive_ignore:E3, naive_ignore:E6, naive_ignore:E8, naive_ignore:E7, drop:E2, drop:E7, collapse_sets |
| a3 | verdict | {"verdict": "undetermined"} | regret<=0 | 1.00 | skip:defect, skip:recompute, skip:weights, skip:bar, B_prior, naive_ignore:E2, naive_ignore:E3, naive_ignore:E6, naive_ignore:E8, naive_ignore:E7, drop:E2, drop:E7 |

## Rivals

| rival | score | killed with margin on | gated |
|---|---|---|---|
| skip:defect | 0.00 | a1, a2, a3 | yes |
| skip:recompute | 0.00 | a1, a2, a3 | yes |
| skip:weights | 0.33 | a2, a3 | yes |
| skip:bar | 0.67 | a3 | yes |
| B_prior | 0.00 | a1, a2, a3 | yes |
| naive_ignore:E2 | 0.00 | a1, a2, a3 | yes |
| naive_ignore:E3 | 0.00 | a1, a2, a3 | yes |
| naive_ignore:E6 | 0.00 | a1, a2, a3 | yes |
| naive_ignore:E8 | 0.00 | a1, a2, a3 | yes |
| naive_ignore:E7 | 0.33 | a2, a3 | yes |
| drop:E2 | 0.00 | a1, a2, a3 | yes |
| drop:E7 | 0.00 | a1, a2, a3 | yes |
| collapse_sets | 0.67 | a2 | yes |

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
- G11_depth: **pass**
- G12_nuisance: **pass**
- G13_anti_prior: **pass**
- G14_witness: **pass**
- G15_plan: **pass**
- G16_audit: **pass**
- G17_oracle_bias: **pass**

## Human checklist (tick each; any 'no' -> reject or fix the blueprint)

- [ ] The notebook and lab notes read like a plausible team artefact; nothing in them states the answer.
- [ ] Each question is unambiguous given manual.md Section 3 (point vs. set vs. verdict semantics).
- [ ] Each known unknown is truly undeterminable in this lab (see cert.json G4) and its documented range is the one used for the key.
- [ ] The keys follow from the cards' stated forms (spot-check one numeric key by hand from hidden/world.json).
- [ ] The oracle's runs (solution/oracle_runs.jsonl) are a design a competent researcher could think of.
- [ ] The rival that encodes the notes' reading fails for the reason the design note says.
- [ ] Grounding references for each active card are appropriate (cards.py).
- [ ] For a counterexample item: the free parameters are the only ones that could flip the claim, and the oracle's witness is not the only one that works.
- [ ] For a plan item: every world in the world set is genuinely consistent with the notebook, and its label is the conclusion a careful analyst would reach.
- [ ] For an audit item: the script reads as a teammate's work, the defect is silent (it runs and prints a plausible number), and every red herring is provably harmless.

Reviewer notes:

