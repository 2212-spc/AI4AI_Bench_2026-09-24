# Ship the nightly classifier on the full corpus

The ranking team trains a small classifier - a numpy MLP, `/app/repo/train.py`, hyper-parameters in
`/app/repo/config.json` - on crowd-labelled data.  Crowd labels are noisy (a sizeable fraction are simply
wrong); quality is measured on expert (gold) labels.

Everything so far was developed on a **5% uniform sample of the corpus**: `/app/data/sample.npz`
(4000 rows, crowd labels).  Hyper-parameters were chosen by dev accuracy on `/app/data/dev.npz` (1000 rows, gold
labels), training on the sample for **1000 steps** (batch 128, i.e. 32 passes over the sample).  The team's
sweep is in `/app/data/tune_log.csv`; `config.json` is its best row.

**Production** trains on the **full corpus: 80000 rows** (the sample is part of it - same sources, same crowd,
same label noise) for **20000 steps** - the same 32 passes over the data.  From the shipped `/app/repo` it runs

    python3 train.py --data corpus.npz --steps 20000 --predict test_X.npy --out preds.npy --seed 0

and the same with `--seed 1`, then scores both on a held-out **gold-labelled test set** (20000 rows, same
distribution as `dev.npz`).  The **production score** is the mean test accuracy of the two runs.  You cannot
access the corpus or the test set.

**Goal: production score >= 0.735.**

## Rules of the production machine

* One CPU core, Python 3 + numpy only, no network.  **CPU-time limit 120 s per run** (the current job needs
  about 20 s); memory 2 GB.
* The job may read only its command-line inputs and files inside the repo.  Do not ship data or fitted
  parameters in the repo (no rows of the sample or dev set, no weights trained on them); code and
  hyper-parameters are yours to change.  Repo size limit 1 MB.
* You may change anything in `/app/repo` - `config.json`, `train.py`, add modules - as long as the command
  above works unchanged and writes one integer class per test row.

## Deliverables

1. **`/app/repo`** - graded by running it in production as described (pass = score >= 0.735).
2. **`/app/report.md`** - a short note to the team: what you changed and why.  Not scored.
