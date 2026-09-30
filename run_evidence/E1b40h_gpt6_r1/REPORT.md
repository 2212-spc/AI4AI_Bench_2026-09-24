# Recipe change decision

Ship all six changes: `c1,c2,c3,c4,c5,c6`.

I used all 40 available runs, evaluating the main candidate arms on the same four
seeds (101–104). The four-seed means were:

| arm | mean validation MSE |
|---|---:|
| baseline | 0.05013 |
| c1,c2,c3,c4 | 0.04921 |
| c1,c3,c4,c6 | 0.04942 |
| c2,c3,c4,c6 | 0.05018 |
| c1,c2,c3,c4,c6 | 0.05361 |
| c1,c2,c3,c4,c5,c6 | **0.04760** |

The proposed teammate bundle `c1,c2,c4,c6` diverged on all four common seeds,
and adding c5 without c3 had the same failure. Since divergence counts as MSE
0.3466, those combinations are clearly unsafe. Adding c3 and c5 together
rescued the high-learning-rate/high-momentum bundle: the full six-change arm was
stable on every seed and had the lowest observed mean. This comparison also
shows why single-change results and greedy selection are misleading here.

The shipped recipe's predicted expected validation MSE is **0.0476**, based on
the matched-seed mean for the full arm.
