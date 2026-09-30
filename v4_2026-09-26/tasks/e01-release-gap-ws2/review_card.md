---
task: e01-release-gap-ws2
blueprint: e01-release-gap
world_seed: 2
review_status: pending
reviewer: 
---

# Review card: Release gap: the denominator, the contamination, the error bar and the weighting of one eval report

**Claim tested.** hold an evaluation report to the convention it will be released under: separate the reported denominator from the quantity of interest, measure contamination instead of trusting a detector score, price the between-call variance a binomial error bar omits, and report a weighting-dependent comparison as undetermined

**Lab.** evallab

**Cards.** E1 (Two-parameter item response), E2 (Format sensitivity and answer-extraction failure), E3 (Contamination through duplicated items), E6 (Between-call variance and the resulting detectable effect size), E7 (Slice mixture and aggregation convention (Simpson)), E8 (n-gram contamination detector with calibration error)

**Obstacles.** O4, O16, O10, O5

**Design note.** E01 release gap: an eval team is about to publish "M_new beats M_ref by several points".  Four separate
things are wrong with that sentence, and each one is a measurement the agent has to make.

(O4 measurement artifact, O16 denominator swap that changes the sign, O10 aggregation, O5 non-identifiability.)

The team's memo reports `acc` exactly as the service returns it: correct / *extractable* items.  The release
harness counts an unparseable answer as wrong, and M_new's answers are harder to parse than M_ref's (E2,
`ext_model_sd`), so the convention is not a rounding detail - it moves each model by a different amount
(O16).  A quarter of the bank occurs in the pretraining corpus and M_new memorised some of it (E3); the
team's evidence against contamination is the *cheap* detector, whose score is a monotone function of the
occurrence count plus item-level noise (E8), so thresholding it neither finds the duplicated items nor
bounds their effect - only the expensive exact scan does.  The memo's error bar is the binomial one; the
service also has a between-call shift shared by every item in a call (E6), which the binomial formula
cannot see and which no number of items removes, only repetitions (Miller 2024).  Finally the release
report will weight slice 0 by a factor the product team has not fixed (known unknown, zero footprint), and
the two models' slice profiles differ (E7 `slice_off_sd`), so the headline comparison is genuinely
undetermined.

Note on E7: this task does *not* use aggregation as a difficulty lever - the card's own grounding records
that micro-vs-macro is a weak one.  The weighting enters as an *ignorance* mechanism: an undecided release
weight turns one question into a set and one comparison into `undetermined`.  The strong lever here is the
denominator, exactly as the card says.

Useful designs: `corpus mode=scan` over the whole bank (the only sound contamination measurement);
`detail=1` rows, which give per-item outcomes and therefore a clean-minus-all difference measured *inside*
one call, where the shared between-call shift cancels; one high-`reps` row, whose `reps_acc` is a direct
sample of the quantity the binomial formula gets wrong; per-slice rows for the reweighting.  `by_slice` and
`detail` are first-repetition only, so per-item resolution costs one request per replicate while a mean
does not.

## Items, keys, tolerances

| id | kind | key | tol | oracle pass (fresh reps) | rivals killed with margin |
|---|---|---|---|---|---|
| q1 | point | [0.4445, 0.4445] | 0.0229 | 1.00 | skip:denominator, skip:clean, skip:detector, naive_ignore:E2, naive_ignore:E8, drop:E2, widen_points |
| q2 | point | [0.1166, 0.1166] | 0.0216 | 1.00 | skip:clean, skip:detector, naive_ignore:E8, drop:E3, widen_points |
| q3 | set | [0.5332, 0.6229] | 0.0123 | 1.00 | skip:denominator, skip:slices, naive_ignore:E2, naive_ignore:E7, drop:E2, drop:E3, drop:E7, collapse_sets |
| q4 | point | [0.05273, 0.05273] | 0.0112 | 1.00 | B_prior, drop:E6, widen_points |
| q5 | verdict | {"verdict": "undetermined"} | regret<=0 | 1.00 | skip:denominator, skip:clean, skip:detector, skip:slices, naive_ignore:E2, naive_ignore:E7, naive_ignore:E8, drop:E2, drop:E7 |

## Rivals

| rival | score | killed with margin on | gated |
|---|---|---|---|
| B_prior | 0.80 | q4 | yes |
| skip:denominator | 0.40 | q1, q3, q5 | yes |
| skip:clean | 0.40 | q1, q2, q5 | yes |
| skip:detector | 0.40 | q1, q2, q5 | yes |
| skip:slices | 0.60 | q3, q5 | yes |
| naive_ignore:E2 | 0.40 | q1, q3, q5 | yes |
| naive_ignore:E7 | 0.60 | q3, q5 | yes |
| naive_ignore:E8 | 0.40 | q1, q2, q5 | yes |
| drop:E2 | 0.40 | q1, q3, q5 | yes |
| drop:E3 | 0.40 | q2, q3 | yes |
| drop:E6 | 0.80 | q4 | yes |
| drop:E7 | 0.60 | q3, q5 | yes |
| collapse_sets | 0.80 | q3 | yes |
| widen_points | 0.40 | q1, q2, q4 | yes |

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

