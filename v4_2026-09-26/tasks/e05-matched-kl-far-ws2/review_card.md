---
task: e05-matched-kl-far-ws2
blueprint: e05-matched-kl-far
world_seed: 2
review_status: pending
reviewer: 
---

# Review card: Matched-KL ship review under a horizon the service will not run

**Claim tested.** separate what a proxy reward can settle from what only a gold evaluation can, put two methods at the same distance from the initial policy when neither knob states that distance, and commit to a decision rule that holds in every world the disclosed evidence leaves open

**Lab.** rllab

**Cards.** R1 (Reward-model overoptimization), R2 (KL budget as the invariant, not the knob that sets it), R4 (Length as the dominant reward channel)

**Obstacles.** O13, O4, O1, O11, O17

**Design note.** E05 matched-KL ship review with a horizon the service will not run: e04's world and questions, plus the
two levers the e04 measurements showed were the ones actually missing.

(O13 plan-dependence, O4 measurement artifact, O1 confounded notebook, O11 under-determination,
O17 unreachable configuration.)

**Why this variant exists.**  Twenty frontier runs on e04 (2026-09-26, gpt-6-astra and claude-fable-5-1,
two repetitions each on four blueprints) passed 18 of 20 instances, and the per-item margins said why:
on e04's q1 - the item carrying a declared anti-prior gap of 51 tolerances - all six runs answered with
*zero* error.  The ledger shows how: `svc=train` reports `kl` without noise, so the question "what n
reaches the same KL as this PPO recipe" is a one-dimensional search on a reported field, and the runs
simply bisected it (130 best-of-n rows in one case).  A declared prior gap cannot make a question hard
when nobody has to use the prior.  Chain depth and nuisance count were no better: both are properties of
the *intended* derivation, and measuring the answer directly bypasses the derivation entirely.

So this variant changes exactly two things and nothing else:

  * **q5 asks the same question about a configuration the lab refuses to run.**  The `steps` knob stops at
    STEPS_MAX and the team's plan of record runs past it, so no request measures that policy.  The route
    is to identify the approach law kl(S) = kl_knob * (1 - exp(-S/steps0)) from inside the permitted range
    and extrapolate.  q1 is deliberately left as it was, so (q1, q5) is a matched pair on reachability
    within one instance, one world, one set of observables and one tolerance.
  * **The caps sit at about 1.35x the reference solution instead of 2.6x.**  On e04 the measured usage was
    79 / 110 / 156 / 177 / 217 / 240 lab runs against a reference that spends 95; four of six runs would
    not fit here, which is the point - the slack is what paid for the search.

Everything below this paragraph is e04, and the rest of this docstring describes it unchanged.

**What the e05 measurements then said (2026-09-26, four runs on ws=2, two models x two repetitions).**
The variant answered its own question in the negative.  q5 forced the derivation - gpt-6-astra wrote "the
PPO law matches an exponential approach, and the matched sampling budgets are near the exact integer
crossings n=270 for the 945-step proposal and n=2163 for the 2160-step continuation" - and then all four
runs answered it *exactly*, 0.00 tolerances, indistinguishable from reachable q1.  The caps did not bind
either: two of the four finished at exactly the 130-request ceiling and still answered correctly.  So
unreachability changes the route and not the difficulty, at least while the law is smooth, low-dimensional
and reported without noise.

The one item that did separate the two models is q2, which is not structural at all: gpt-6-astra missed it
by 4.16 and 4.06 tolerances in both repetitions while fable-5-1 passed at 0.10 and 0.64.  Its route is
recovered in `_q2_slope_routes` - the widest secant the build sells, in place of the extrapolation to zero
distance - and is now registered as a rival so no future world can ship with a tolerance that forgives it.
The lever that works is therefore an estimator that has to be *designed* under noise and inverted through a
saturating response, not a longer chain, a bigger prior gap, more nuisance mechanisms, or a tighter budget.

(One earlier reading of these runs put q5 at 0.44 T rather than 0.00.  That was this file's own defect, not
the models': the matched budget solves to 2162.6907 while a sampling budget is a count, and carrying the
fraction into the key made every correct answer - the reference one included - miss by the rounding.  See
the note on q5's key in `items`, and gate G17, which now fails any item whose reference error is an offset
that bought its own tolerance.)

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
| q1 | point | [270, 270] | 0.5 | 1.00 | B_prior, skip:bon_identity, skip:reach, naive_ignore:R2, B_guess:geo, B_guess:ari, drop:R2, widen_points |
| q2 | point | [0.3923, 0.3923] | 0.0433 | 1.00 | skip:separate, skip:len_pen, skip:baseline, skip:extrapolate, B_slope:local, B_slope:span, naive_ignore:R4, drop:R1, drop:R4, widen_points |
| q3 | prereg | {"labels": ["bon", "ppo", "undetermined"]} | regret<=0 | 1.00 | naive_ignore:R1, skip:gold, plan:underconverged, plan:underpowered, drop:R2 |
| q4 | decision | {"choice": "bon_b", "regret": {"ppo_ship": 0.09987735551134114, "ppo_light": 0.9092480666806922, "bon_a": 0.8978488391227488, "bon_b": 0.0}} | regret<=0.0333 | 1.00 | naive_ignore:R1, skip:gold, drop:R1, drop:R2 |
| q5 | point | [2163, 2163] | 0.5 | 1.00 | skip:bon_identity, naive_ignore:R2, skip:extrapolate_far, skip:saturation, B_far:linear, B_far:proportional, drop:R2, widen_points |

## Rivals

| rival | score | killed with margin on | gated |
|---|---|---|---|
| B_prior | 0.80 | q1 | yes |
| skip:bon_identity | 0.60 | q1, q5 | yes |
| skip:reach | 0.80 | q1 | yes |
| naive_ignore:R2 | 0.60 | q1, q5 | yes |
| B_guess:geo | 0.80 | q1 | yes |
| B_guess:ari | 0.80 | q1 | yes |
| skip:extrapolate_far | 0.80 | q5 | yes |
| skip:saturation | 0.80 | q5 | yes |
| B_far:linear | 0.80 | q5 | yes |
| B_far:proportional | 0.80 | q5 | yes |
| skip:separate | 0.80 | q2 | yes |
| skip:len_pen | 0.80 | q2 | yes |
| skip:baseline | 0.80 | q2 | yes |
| skip:extrapolate | 0.80 | q2 | yes |
| B_slope:local | 0.80 | q2 | yes |
| B_slope:span | 0.80 | q2 | yes |
| naive_ignore:R4 | 0.80 | q2 | yes |
| naive_ignore:R1 | 0.60 | q3, q4 | yes |
| skip:gold | 0.60 | q3, q4 | yes |
| plan:underconverged | 0.80 | q3 | yes |
| plan:underpowered | 0.80 | q3 | yes |
| drop:R1 | 0.60 | q2, q4 | yes |
| drop:R2 | 0.00 | q1, q3, q4, q5 | yes |
| drop:R4 | 0.80 | q2 | yes |
| widen_points | 0.40 | q1, q2, q5 | yes |

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

