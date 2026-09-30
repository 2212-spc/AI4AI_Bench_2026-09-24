# MiniLab recipe decision

Ship **c1, c2, c3, c5, c6**; retain the original hidden width (omit c4).
Estimated expected validation MSE: **0.041374**, the arithmetic mean of the
three runs of exactly this recipe.

## Evidence and budget

Used exactly **80/80 runs**: all 64 subsets at seed 1001, then the eight
lowest-MSE subsets at fresh seeds 1002 and 1003. Seeds were shared across
configurations to support paired comparisons; changing width or the RNG stream
can change how those seeds translate into initialization and data order.
Divergence is scored as **0.3466**, never discarded or treated as missing.
The teammate's seed-9 notes motivated checking interactions but were not pooled
into the final estimate.

| Changes | Seed 1001 | Seed 1002 | Seed 1003 | Mean |
|---|---:|---:|---:|---:|
| c1, c2, c3, c5, c6 | 0.040978 | 0.041840 | 0.041304 | 0.041374 |
| c1, c2, c3, c5 | 0.041504 | 0.041058 | 0.041621 | 0.041394 |
| c1, c2, c3 | 0.041688 | 0.044126 | 0.042374 | 0.042729 |
| c2, c5 | 0.042525 | 0.042927 | 0.042743 | 0.042732 |
| c1, c5 | 0.043731 | 0.042420 | 0.042739 | 0.042963 |
| c1, c2, c3, c6 | 0.043692 | 0.043464 | 0.041925 | 0.043027 |
| c2 | 0.042862 | 0.043352 | 0.043045 | 0.043086 |
| c1, c5, c6 | 0.043732 | 0.044450 | 0.043461 | 0.043881 |

## Why this combination

- **Clipping changes the conclusion from single-edit ablations.** At seed 1001,
  c1+c2 scored 0.072554; adding c3 reduced it to 0.041688. With c5+c6 also
  enabled, clipping reduced MSE from 0.052299 to 0.040978. Clipping alone
  scored 0.051517 against baseline 0.049733: its value is conditional on the
  more aggressive optimizer.
- **The proposed wider combination is unsuitable.** The teammate's
  c1+c2+c4+c6 proposal diverged at seed 1001, as did all four unclipped
  combinations containing c1+c2+c4. Even with clipping and weight decay,
  adding c4 to the selected recipe increased MSE from 0.040978 to 0.047740
  in the full screen. This supports keeping width 96.
- **Weight decay matters in the selected context.** Adding c5 to c1+c2+c3+c6
  improved all three paired runs, reducing mean MSE from 0.043027 to 0.041374
  (3.8%). With c6 absent, the corresponding mean reduction was from 0.042729
  to 0.041394 (3.1%). Its small standalone effect misses this interaction.
- **c6 is a statistical tie.** Its paired effects in the leading recipe have
  mixed signs; the mean difference is only 0.000020. I retain it because that
  exact subset has the lowest observed mean, without claiming a demonstrated
  population-level advantage over c1+c2+c3+c5.

## Estimation limits

The selected recipe's sample standard deviation is 0.000435 and its descriptive
standard error is 0.000251 (three seeds). There were no divergences among its
three runs, or among any of the 16 follow-up runs. These observations do not
rule out rare failures. The estimate uses all three observations rather than
the best single run; the mean on the two fresh seeds alone is 0.041572, close
to the reported prediction.

All subsets were screened, but 56 received only one seed. Selection on seed
1001 and only three observations per finalist limit certainty about the global
optimum and make the standard error an incomplete measure of uncertainty.
The evidence supports the selected configuration and its near-tied variant;
it does not establish their ordering conclusively.

Full results are in `experiments.json`; the cluster's authoritative run log
is saved in `lab_history.json`.
