# Task: estimate the effect of launching a seller feature to everyone

You are the experimentation analyst for an online marketplace. A new seller-side feature was tested
in an A/B experiment and the launch decision needs one number: **how much would the average seller's
outcome (sales in the experiment period) change if the feature were launched to every seller,
compared with launching it to nobody?**

## The experiment
- The marketplace is split into independent **markets** (city x category). Sellers in the same market
  sell to the same pool of buyers; markets do not share buyers.
- The experiment used a two-stage design. Stage 1: each market was assigned a **treated fraction**
  `p_market`, drawn at random from levels between 0.1 and 0.9 (never 0 or 1). Stage 2: inside each
  market, exactly `round(p_market * n_sellers)` sellers, chosen uniformly at random, got the feature.
- For every seller you also have the same outcome measured in a pre-period, before the experiment
  (nobody had the feature then).

## What you edit
`solution.py` must define

```python
def estimate(units: pd.DataFrame) -> float
```

`units` has one row per seller of one experiment, columns:
`market` (int id, arbitrary), `p_market` (float, the design fraction of that market), `treated` (0/1),
`pre_outcome` (float, pre-period outcome), `outcome` (float, outcome during the experiment).
Return the estimated **global treatment effect** in outcome units:

    GTE = (1/N) * sum over all N sellers in the experiment of
          [ outcome_i if every seller in every market were treated  -  outcome_i if no seller were treated ]

(the same markets, sellers and period as the data; average per seller, so large markets count more).
Only `solution.py` is graded; you may add helper files next to it. numpy / pandas / scipy /
scikit-learn only, CPU, no network; the 8 experiments of a setting run in one process with a 600 s total limit.

## What you have
- `data/dev_experiments.pkl` - list of 2 dev settings, each a dict with `name`, `units` (list of 8
  replicate experiments, DataFrames as above), `gte` (true GTE of each replicate), `base` (average
  seller outcome if nobody were treated, per replicate). Load with `pickle`.
- `baselines/` - standard analyses with the same interface: `diff_in_means.py`, `cuped.py`
  (CUPED with the pre-period outcome), `reg_adjust.py` (OLS with treatment, pre-period outcome and
  market fixed effects). `solution.py` starts as a copy of `diff_in_means.py`.
- `eval_dev.py` - `python3 eval_dev.py [file.py]` scores a solution on the dev settings.

## How it is graded
A setting is a marketplace world; each setting is evaluated on 8 independent replicate experiments.
Metric per setting = RMSE over the replicates of (estimate - true GTE) / base.
**Pass = metric <= 0.04 on every setting.** Graded settings = the 2 dev settings + **4 hidden
settings**. The hidden settings differ from the dev ones in the number of markets (fewer or more),
in how unequal market sizes are, in noise, in the size and direction of the feature's effects
(including how sellers' outcomes depend on each other, if at all), and in the design (which treated
fractions were used and how often). An estimator that is right on average but breaks in one world fails.

Write a short `NOTES.md`: what structure you found in the data, what your estimator assumes, and why
you expect it to hold in the hidden settings.
