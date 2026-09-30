---
task: e04-matched-kl-ws5
blueprint: e04-matched-kl
world_seed: 5
review_status: pending
reviewer: 
---

# Review card: Matched-KL ship review: pre-register the experiment that decides best-of-n against PPO

**Claim tested.** separate what a proxy reward can settle from what only a gold evaluation can, put two methods at the same distance from the initial policy when neither knob states that distance, and commit to a decision rule that holds in every world the disclosed evidence leaves open

**Lab.** rllab

**Cards.** R1 (Reward-model overoptimization), R2 (KL budget as the invariant, not the knob that sets it), R4 (Length as the dominant reward channel)

**Obstacles.** O13, O4, O1, O11

**Design note.** E04 matched-KL ship review: an alignment team has to choose between best-of-n and PPO at the same
distance from the initial policy, and has to commit to the experiment *before* seeing its result.

(O13 plan-dependence, O4 measurement artifact, O1 confounded notebook, O11 under-determination.)

The lab has one reward model, frozen at deploy time, so the only live mechanisms are R1 (proxy versus
gold), R2 (how each method reaches a given KL) and R4 (length).  That is deliberate: every question here
is about what the team's own proxy number *cannot* tell them, and a third source of variation would let an
agent blame the disagreement on the reward model instead.

**q3 is the item this blueprint exists for.**  Best-of-n and PPO reach the same KL by different routes, and
at a matched KL they produce **exactly the same proxy reward** - the proxy is a function of d = sqrt(KL) and
of mean length, and mean length is itself a function of d and the length penalty, so the method does not
enter it at all.  The gold win rate *does* differ, because the two methods pay for distance differently:
best-of-n pays a term in d^2 and policy gradient a term in d*log(d), and which of them is cheaper at a
given d depends on how fast the proxy and the gold diverge - a constant the disclosed notebook contains no
information about, because no train observable depends on it.  Three worlds are therefore consistent with
everything the team disclosed and they disagree about which method wins.  A plan that compares proxy
rewards returns the same number in all three; a plan that buys gold comparisons separates them, but only if
it buys enough of them.  That is the whole point of the pre-registered form: the agent cannot measure first
and then write a rule that happens to be right where it is standing.

q1 is the anti-prior item, and it is the memo's own mistake twice over.  The memo dismisses best-of-n by
quoting the textbook KL = log(n) and by reading its own `kl=` knob as the KL the run reaches.  Both are
wrong here: the exact best-of-n divergence is log(n) - (n-1)/n, and a PPO run approaches its target over
steps and stops short of it.  Correcting one and not the other still gives the wrong n, and the three wrong
answers are hundreds of tolerances away from the right one.

q2 is the length confound.  The team's sweep varied the KL target only, so mean length and proxy reward
move together and the memo reads the whole gain as quality.  The instrument that separates them is
`len_pen`, which moves length at a *fixed* KL - but the length term is measured against the length at zero
distance, which is not the shortest run in the notebook, and the penalty is capped, so the quality-only
limit has to be extrapolated rather than read off the largest penalty available.

q4 is the same proxy-versus-gold confusion in its cheapest form: four candidate recipes, where the one with
the best proxy reward is never the one with the best gold win rate.

## Items, keys, tolerances

| id | kind | key | tol | oracle pass (fresh reps) | rivals killed with margin |
|---|---|---|---|---|---|
| q1 | point | [72, 72] | 0.5 | 1.00 | B_prior, skip:bon_identity, skip:reach, naive_ignore:R2, B_guess:geo, B_guess:ari, drop:R2, widen_points |
| q2 | point | [0.3504, 0.3504] | 0.0364 | 1.00 | skip:separate, skip:len_pen, skip:baseline, skip:extrapolate, naive_ignore:R4, drop:R1, drop:R4, widen_points |
| q3 | prereg | {"labels": ["bon", "ppo", "undetermined"]} | regret<=0 | 1.00 | naive_ignore:R1, skip:gold, plan:underconverged, plan:underpowered, drop:R2 |
| q4 | decision | {"choice": "bon_b", "regret": {"ppo_ship": 0.6824799718342149, "ppo_light": 0.898274593575724, "bon_a": 0.6980864701495197, "bon_b": 0.0}} | regret<=0.227 | 1.00 | naive_ignore:R1, skip:gold, drop:R1, drop:R2 |

## Rivals

| rival | score | killed with margin on | gated |
|---|---|---|---|
| B_prior | 0.75 | q1 | yes |
| skip:bon_identity | 0.75 | q1 | yes |
| skip:reach | 0.75 | q1 | yes |
| naive_ignore:R2 | 0.75 | q1 | yes |
| B_guess:geo | 0.75 | q1 | yes |
| B_guess:ari | 0.75 | q1 | yes |
| skip:separate | 0.75 | q2 | yes |
| skip:len_pen | 0.75 | q2 | yes |
| skip:baseline | 0.75 | q2 | yes |
| skip:extrapolate | 0.75 | q2 | yes |
| naive_ignore:R4 | 0.75 | q2 | yes |
| naive_ignore:R1 | 0.50 | q3, q4 | yes |
| skip:gold | 0.50 | q3, q4 | yes |
| plan:underconverged | 0.75 | q3 | yes |
| plan:underpowered | 0.75 | q3 | yes |
| drop:R1 | 0.50 | q2, q4 | yes |
| drop:R2 | 0.00 | q1, q3, q4 | yes |
| drop:R4 | 0.75 | q2 | yes |
| widen_points | 0.50 | q1, q2 | yes |

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

