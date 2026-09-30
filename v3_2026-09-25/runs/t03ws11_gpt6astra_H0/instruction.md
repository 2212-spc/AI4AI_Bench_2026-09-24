# Task: Quality filtering for a data-constrained production run

You are a research engineer in a pretraining team.  The team's notebook (`/app/notebook/`) holds earlier runs and the team's notes.  You have a compute budget on the team's simulated lab (`lab` command; read `/app/manual.md` first - it defines the lab, its guarantees and how answers are graded).

Answer every question below by writing `/app/answers.json` (a template with the right keys is already there).  You may run any experiments within the budget.  Work carefully: the notes may be wrong, and some questions may not be fully answerable from anything the lab can measure - the manual says how to answer those.

## Questions

- **q1** Repetition cost: by how much (nats/token) does repetition raise the final loss of the unfiltered (q=0) production run N=1e9, D=2e11 on a snapshot of 4e10 raw unique tokens, compared with the same run on unlimited unique data?  (A positive number; 0 if repetition were free.)  [unit: nats] Answer format: `{"lo": "number", "hi": "number"}`
- **q2** Filtering effect: final loss of the production run (N=1e9, D=2e11, snapshot of 4e10 raw unique tokens) at q=0.5 minus its final loss at q=0, in nats/token (negative = filtering helps).  [unit: nats] Answer format: `{"lo": "number", "hi": "number"}`
- **q3** For the production run N=1e9, D=2e11 on a snapshot of 4e10 raw unique tokens, which filter level gives the lowest final loss?  Options: A: q=0, B: q=0.3, C: q=0.6, D: q=0.85  Answer format: `{"choice": "one of A, B, C, D"}`
- **q4** With enough unique data that nothing is repeated, filtering at q=0.6 lowers the final loss of N=1e9 trained on D=1e11 tokens by at least 0.050 nats (compared with q=0).  Answer format: `{"verdict": "supported | refuted | undetermined"}`
- **q5** Filtering effect on the next snapshot as it will be: final loss of N=1e9, D=2e11 at q=0.5 minus at q=0, in nats/token (negative = filtering helps); the snapshot size is the documented known unknown.  [unit: nats] Answer format: `{"lo": "number", "hi": "number"}`
- **q6** For the production run as planned (N=1e9, D=2e11, next snapshot), filtering at q=0.6 gives a lower final loss than no filtering.  Answer format: `{"verdict": "supported | refuted | undetermined"}`

When you are done, make sure `/app/answers.json` is valid JSON with one entry per question id.
