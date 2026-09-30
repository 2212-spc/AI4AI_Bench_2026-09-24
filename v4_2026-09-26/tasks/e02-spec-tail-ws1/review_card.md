---
task: e02-spec-tail-ws1
blueprint: e02-spec-tail
world_seed: 1
review_status: pending
reviewer: 
---

# Review card: Speculative tail: what a proposal length you cannot run would buy, and what one replica delivers

**Claim tested.** separate a speculative decoder's yield from its cost, price a proposal length the build cannot run as a question about what the evidence leaves open rather than as a measurement, and read a production replica's capacity off the pool it actually reserves

**Lab.** servelab

**Cards.** S1 (Roofline: arithmetic intensity and the critical batch for decoding), S3 (Speculative decoding with positionally decaying acceptance), S5 (KV-cache capacity sets the feasible batch)

**Obstacles.** O11, O4, O2, O13

**Design note.** E02 speculative tail: an inference team has to decide whether to ship a draft scheduler that raises the
proposal length from 4 to 8, and what one production replica will actually deliver at its capacity ceiling.

(O11 extrapolation past the measured region, O4 measurement artifact, O2 confounded attribution,
O13 regime change.)

The build installed in the lab caps `spec_g` at 4, so every acceptance rate the team can measure lives at
positions 0-3.  The claim they want to make is about positions 4-7.  S3's decay beyond `tail_from` is a
*separate* constant (`rho_tail`), it is a documented known unknown, and nothing runnable touches it - so
"does proposal length 8 beat 4" is not a measurement, it is a question about which continuations the
evidence leaves open.  That is what the two counterexample questions ask, from the two sides: one claim is
true in this world but *not entailed* (a slower tail reproduces every number the team has and reverses it),
the other is a bound that **is** entailed, and the analyst who says "unmeasurable, therefore unbounded" is
wrong.  The witness for the first has to move the acceptance constants too - moving `rho_tail` alone leaves
the measured positions where they are and the evidence still points the other way - so it lives in the thin
diagonal band the profile fit leaves open, a few per cent of the declared box.

q1 is the confound.  Every aggregate the memo quotes mixes two things: how much *yield* speculation buys
(the acceptance profile) and what it *costs* (the draft model's time per proposed token).  The memo
attributes the whole of the observed speedup to acceptance.  Separating them needs a matched baseline (the
memo's is at a different context and batch, where the decode step itself is a different length), the
per-position profile rather than the reported mean `accept_rate`, and the observation that `ms_per_token`
is per *accepted output token*, not per verification step - so the step-time ratio is `(ms4/ms0) * E4`.
Skip that last step and the arithmetic hands back a *negative* drafting cost; noticing the contradiction is
the point.  The published c = 0.128 (Leviathan 2023) is not this lab's constant.

q2 is the capacity question and the one place S1 and S5 interact.  The production replica reserves a fixed
KV pool - smaller than the free memory on the team's dev box, exactly as a serving stack with a
`gpu_memory_utilization` setting does - and fills it with whole sequences of the p95 context length.  That
length is undecided (the second known unknown), so the answer is a set; at the short end the replica is
past the critical batch and sits on the compute roof, at the long end it is memory-bound and well under it.
Three shortcuts each collapse it: reading the lab box's own `batch_max` instead of the production pool,
applying the memory-traffic formula everywhere and missing the roof, and answering at the midpoint as if
the range did not matter.

Useful designs: a matched `spec_g=0` / `spec_g=4` pair at one `(seq, batch)` - the decode step cancels out
of the ratio only if both sides are at the same configuration; long `dur` on those rows, since the
acceptance positions are estimated from the proposals the run had time to make; `bench` at the two
production points directly, which is possible precisely because the reserved pool is smaller than the free
memory the lab reports; and `batch_max` / `kv_bytes_per_seq`, which are exact and cost whatever the
shortest run costs.

## Items, keys, tolerances

| id | kind | key | tol | oracle pass (fresh reps) | rivals killed with margin |
|---|---|---|---|---|---|
| q1 | point | [0.1841, 0.1841] | 0.0113 | 1.00 | B_prior, skip:matched, skip:profile, skip:perstep, naive_ignore:S3, drop:S3, widen_points |
| q2 | set | [5363, 7841] | 141 | 1.00 | skip:capacity, skip:roofline, skip:extremes, naive_ignore:S5, naive_ignore:S1, drop:S1, drop:S5, collapse_sets |
| q3 | cex | {"verdict": "refutable"} | regret<=0 | 1.00 | skip:profile, skip:tail, naive_ignore:S3, drop:S3 |
| q4 | cex | {"verdict": "entailed"} | regret<=0 | 1.00 | skip:profile, skip:bound, naive_ignore:S3, drop:S3 |

## Rivals

| rival | score | killed with margin on | gated |
|---|---|---|---|
| B_prior | 0.75 | q1 | yes |
| skip:matched | 0.75 | q1 | yes |
| skip:profile | 0.25 | q1, q3, q4 | yes |
| skip:perstep | 0.75 | q1 | yes |
| skip:capacity | 0.75 | q2 | yes |
| skip:roofline | 0.75 | q2 | yes |
| skip:extremes | 0.75 | q2 | yes |
| skip:tail | 0.75 | q3 | yes |
| skip:bound | 0.75 | q4 | yes |
| naive_ignore:S3 | 0.25 | q1, q3, q4 | yes |
| naive_ignore:S5 | 0.75 | q2 | yes |
| naive_ignore:S1 | 0.75 | q2 | yes |
| drop:S1 | 0.75 | q2 | yes |
| drop:S3 | 0.25 | q1, q3, q4 | yes |
| drop:S5 | 0.75 | q2 | yes |
| collapse_sets | 0.75 | q2 | yes |
| widen_points | 0.75 | q1 | yes |

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

