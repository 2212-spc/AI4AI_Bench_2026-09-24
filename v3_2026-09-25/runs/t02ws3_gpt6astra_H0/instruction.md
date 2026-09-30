# Task: Stability of a 7B production run planned from small-scale sweeps

You are a research engineer in a pretraining team.  The team's notebook (`/app/notebook/`) holds earlier runs and the team's notes.  You have a compute budget on the team's simulated lab (`lab` command; read `/app/manual.md` first - it defines the lab, its guarantees and how answers are graded).

Answer every question below by writing `/app/answers.json` (a template with the right keys is already there).  You may run any experiments within the budget.  Work carefully: the notes may be wrong, and some questions may not be fully answerable from anything the lab can measure - the manual says how to answer those.

## Questions

- **q1** log10 of the learning rate above which the production run (N=7e9, D=1.4e11, batch 0.5M, warmup 0.02, qk-layernorm off) diverges.  [unit: log10(lr)] Answer format: `{"lo": "number", "hi": "number"}`
- **q2** log10 of the learning rate above which the same production run diverges with qk-layernorm on.  [unit: log10(lr)] Answer format: `{"lo": "number", "hi": "number"}`
- **q3** The production run exactly as planned in the lab notes (qk-layernorm off, warmup 0.02, peak lr = 0.00143) diverges.  Answer format: `{"verdict": "supported | refuted | undetermined"}`
- **q4** The production run with qk-layernorm on (warmup 0.02) and peak lr = 0.0057 does not diverge.  Answer format: `{"verdict": "supported | refuted | undetermined"}`
- **q5** The production run with qk-layernorm on (warmup 0.02) and peak lr = 0.0299 does not diverge.  Answer format: `{"verdict": "supported | refuted | undetermined"}`
- **q6** Keeping qk-layernorm off and warmup 0.02, which peak learning rate gives the production run the lowest final loss?  A run that diverges counts as the worst possible outcome.  Options: A: 0.000443, B: 0.000554, C: 0.000692, D: 0.000865  Answer format: `{"choice": "one of A, B, C, D"}`
- **q7** With qk-layernorm off but warmup raised to 5% of steps (wu=0.05), the production run at peak lr = 0.00106 does not diverge.  Answer format: `{"verdict": "supported | refuted | undetermined"}`

When you are done, make sure `/app/answers.json` is valid JSON with one entry per question id.
