# Task: Scaling law from intermediate checkpoints, carried to a production run that may be stopped early

You are a research engineer in a pretraining team.  The team's notebook (`/app/notebook/`) holds earlier runs and the team's notes.  You have a compute budget on the team's simulated lab (`lab` command; read `/app/manual.md` first - it defines the lab, its guarantees and how answers are graded).

Answer every question below by writing `/app/answers.json` (a template with the right keys is already there).  You may run any experiments within the budget.  Work carefully: the notes may be wrong, and some questions may not be fully answerable from anything the lab can measure - the manual says how to answer those.

## Questions

- **q1** Among finished runs at the production compute 6*N*D = 1.08e+21 FLOPs (D = C/(6N), same schedule for all), which model size gives the lowest final loss?  Options: A: N=1.2e+09, B: N=2.4e+09, C: N=4.8e+09, D: N=9.6e+09  Answer format: `{"choice": "one of A, B, C, D"}`
- **q2** Final validation loss (nats/token) of the production model N=3e9 trained on D=6e10 tokens with the WSD schedule, run to completion.  [unit: nats] Answer format: `{"lo": "number", "hi": "number"}`
- **q3** The production cosine run (N=3e9, D=6e10) is evaluated at its checkpoint after 40% of its tokens.  By how much (nats/token) is that checkpoint's loss higher than the final loss of a separate WSD run of the same model trained to completion on 0.4*6e10 = 2.4e10 tokens?  [unit: nats] Answer format: `{"lo": "number", "hi": "number"}`
- **q4** If the production cosine run is stopped early (f_stop as documented), by how much (nats/token) will its loss at the stop exceed the final loss it would have reached had it completed?  Give the set over the documented range of f_stop.  [unit: nats] Answer format: `{"lo": "number", "hi": "number"}`
- **q5** The production cosine run's checkpoint after 70% of its tokens has a lower loss than a finished WSD run of the same model trained to completion on 40% of the production tokens.  Answer format: `{"verdict": "supported | refuted | undetermined"}`

When you are done, make sure `/app/answers.json` is valid JSON with one entry per question id.
