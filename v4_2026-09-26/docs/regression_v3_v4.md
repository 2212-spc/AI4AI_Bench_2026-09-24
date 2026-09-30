# v3 -> v4 harness regression

The v4 harness adds four labs, three item forms and six gates on top of v3.  The v3 blueprints are
byte-identical between the two trees (`diff` over all seven files), so any change in their verdicts would
be a change in the harness, not in the content.

Procedure (`/tmp/reg1.py`, run per blueprint to stay inside the shell timeout): build each of the seven v3
blueprints at world seeds 1, 2, 3 under `v3/scalelab` and under `v4/scalelab`, then compare `ok`, every
item key, every item's verification pass rate, every rival's score, and every gate verdict the two
harnesses share.

Result, 21 blueprint x seed pairs:

| blueprint | seeds | identical ok / keys / ver / rival scores | shared-gate verdicts |
|---|---|---|---|
| t01_lr_horizon | 1,2,3 | yes | identical |
| t01b_lr_horizon_tight | 1,2,3 | yes | identical |
| t02_stability_edge | 1,2,3 | yes | identical |
| t03_filter_repeat | 1,2,3 | yes | identical |
| t03b_filter_repeat_tight | 1,2,3 | yes | identical |
| t05_contaminated_benchmark | 1,2,3 | yes | identical |
| t07_checkpoint_ledger | 1,2,3 | yes | identical |

Gates present only in v4: `G11_depth`, `G12_nuisance`, `G13_anti_prior`, `G14_witness`, `G15_plan`,
`G16_audit`.  All six pass vacuously on a blueprint that declares no difficulty and ships no new form,
which is why the v3 blueprints stay green without edits.

Two harness edits were made during e04 development and are covered by this run:

- `build.plan_gate` resolves a callable `NAIVE_PLANS[qid]` against the drawn params (a naive plan names
  runs, and the runs name knob settings the world fixes, so it cannot be a literal).  No v3 blueprint has
  a prereg item, so `plan_gate` iterates an empty list for all 21 pairs.
- `build.plan_gate`'s `n_salt` default rose from 5 to 9.  Same reason: it is only read inside the loop
  over prereg items.
