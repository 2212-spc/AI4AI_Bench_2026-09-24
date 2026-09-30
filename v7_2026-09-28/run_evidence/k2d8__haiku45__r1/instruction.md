# Choose the data mixture for a data-constrained pre-training run

You are the data lead for a pre-training run that will be launched exactly once with a fixed shape:
a **2.5B-parameter** (non-embedding) model trained on **500B tokens**. The training corpus has four domains -
`web`, `code`, `math`, `papers` - each with a fixed, finite amount of unique text (see `/app/docs/corpora.md`).
Your job: choose the sampling mixture for that run, and forecast its final composite evaluation loss so the
planning team can budget follow-up work.

You cannot run the target. You can run small proxy trainings in a simulated lab. Run `/app/bin/lab spec`
first. Each `train` call trains a fresh model with N parameters on D tokens using your mixture and returns its
held-out losses on three eval sets (general, code, math) and their weighted composite. Like real training runs,
proxy results carry run-to-run noise. Proxy compute is limited: 5e+19 FLOPs in total, a run costs 6*N*D.
Example:

    /app/bin/lab train N=1e8 D=2e9 --json '{"mix": {"web": 0.7, "code": 0.1, "math": 0.1, "papers": 0.1}}'

## Deliverable: `/app/mixture.json`

    {"mixture": {"web": 0.0, "code": 0.0, "math": 0.0, "papers": 0.0},
     "predicted_loss": {"lo": 0.0, "hi": 0.0}}

* `mixture` - sampling weights (non-negative, sum to 1). The target run draws `mixture[d] * 500B` tokens from
  domain d; if that is more than the domain's unique tokens, the domain's data is repeated (several epochs).
* `predicted_loss` - an interval for the target run's final composite eval loss (its expected value, without
  run-to-run noise) *for the mixture you submit*. Width at most 0.09.

## How it is graded (mechanically, against the simulator; only the file counts)

1. `mixture.json` is valid (weights, interval, width <= 0.09).
2. The target run's composite loss with your mixture is within **0.0025** of the lowest composite loss any
   mixture can reach for this target run.
3. Your interval contains the target run's composite loss with your mixture.

All three must hold. Your lab history is not graded; spend the budget however you think is best. When you are
done, make sure `/app/mixture.json` is written, then stop.
