# Task: Learning-rate transfer to an over-trained production run (tight budget)

You are a research engineer in a pretraining team.  The team's notebook (`/app/notebook/`) holds earlier runs and the team's notes.  You have a compute budget on the team's simulated lab (`lab` command; read `/app/manual.md` first - it defines the lab, its guarantees and how answers are graded).

Answer every question below by writing `/app/answers.json` (a template with the right keys is already there).  You may run any experiments within the budget.  Work carefully: the notes may be wrong, and some questions may not be fully answerable from anything the lab can measure - the manual says how to answer those.

## Questions

- **q1** log10 of the loss-minimising peak learning rate for N=1e9, D=2e10 tokens, batch 0.5M tokens.  [unit: log10(lr)] Answer format: `{"lo": "number", "hi": "number"}`
- **q2** log10 of the loss-minimising peak learning rate for N=1e9, D=1e12 tokens, batch 0.5M tokens.  [unit: log10(lr)] Answer format: `{"lo": "number", "hi": "number"}`
- **q3** Excess final loss (nats/token) of training N=1e9, D=1e12, batch 0.5M at the notes' planned lr_prod = 0.001453 instead of at the loss-minimising learning rate.  [unit: nats] Answer format: `{"lo": "number", "hi": "number"}`
- **q4** log10 of the loss-minimising peak learning rate for the production run as planned: N=1e9, D=1e12 tokens, batch 4M tokens.  [unit: log10(lr)] Answer format: `{"lo": "number", "hi": "number"}`
- **q5** Holding the number of training tokens fixed (batch 0.5M), making the model 10x larger lowers its loss-minimising learning rate by at least 35%.  Answer format: `{"verdict": "supported | refuted | undetermined"}`
- **q6** A colleague proposes peak lr = 0.00157 for the production run as planned (N=1e9, D=1e12, batch 4M). This value is above that run's loss-minimising learning rate.  Answer format: `{"verdict": "supported | refuted | undetermined"}`
- **q7** Which peak learning rate gives the lowest final loss for N=1e9, D=2e11, batch 0.5M? Options: A: 0.000542, B: 0.00108, C: 0.00217, D: 0.00434  Answer format: `{"choice": "one of A, B, C, D"}`
- **q8** Final validation loss (nats/token) of N=1e9, D=1e12, batch 0.5M trained at its loss-minimising learning rate.  [unit: nats] Answer format: `{"lo": "number", "hi": "number"}`

When you are done, make sure `/app/answers.json` is valid JSON with one entry per question id.
