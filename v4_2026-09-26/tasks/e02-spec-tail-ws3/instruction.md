# Task: Speculative tail: what a proposal length you cannot run would buy, and what one replica delivers

You are a research engineer in an inference-serving team.  The team's notebook (`/app/notebook/`) holds earlier runs and the team's notes.  You have a budget on the team's simulated serving stack (`lab` command; read `/app/manual.md` first - it defines the service, its guarantees and how answers are graded).

Answer every question below by writing `/app/answers.json` (a template with the right keys is already there).  You may run any experiments within the budget.  Work carefully: the notes may be wrong, and some questions may not be fully answerable from anything the lab can measure - the manual says how to answer those.

## Questions

- **q1** With `draft=d_lite` at proposal length 4: the fraction of one verification step's wall time that goes to the draft model rather than to the target model's verification pass.  A fraction in [0, 1]; it is the same at every batch and context length (noise-free expected value).  [unit: fraction] Answer format: `{"lo": "number", "hi": "number"}`
- **q2** Steady-state decode throughput, in output tokens per second, of **one production replica** under `deploy/serving.yaml`: no speculation, `bits=16`, the replica filled with as many whole sequences of the p95 context length as its reserved KV pool holds.  The p95 context length is the first known unknown of Section 5; give the set of values this throughput takes over its documented range (noise-free expected values).  [unit: tokens/s] Answer format: `{"lo": "number", "hi": "number"}`
- **q3** Is the following forced by the evidence?  Free parameters: `a` (acceptance at proposal position 0) in [0.8731, 0.9387], `rho` (the decay over positions 0 to 3) in [0.9294, 0.9695], and `rho_tail` (the decay from position 4 on) in [0.85, 1.00].  A witness gives all three and nothing else.  Answer format: `{"verdict": "refutable | entailed", "witness": {"a": "number in [0.8731, 0.9387]", "rho": "number in [0.9294, 0.9695]", "rho_tail": "number in [0.85, 1]"}}`
    - Claim under test: *Raising the draft scheduler's proposal length from 4 to 8, with the draft model, the hardware and the serving configuration unchanged, raises steady-state decode throughput.*
- **q4** Is the following forced by the evidence?  Free parameters: `a` (acceptance at proposal position 0) in [0.8731, 0.9387], `rho` (the decay over positions 0 to 3) in [0.9294, 0.9695], and `rho_tail` (the decay from position 4 on) in [0.85, 1.00].  A witness gives all three and nothing else.  Answer format: `{"verdict": "refutable | entailed", "witness": {"a": "number in [0.8731, 0.9387]", "rho": "number in [0.9294, 0.9695]", "rho_tail": "number in [0.85, 1]"}}`
    - Claim under test: *Raising the draft scheduler's proposal length from 4 to 8, with everything else unchanged, cannot raise steady-state decode throughput by more than 9.85%.*

When you are done, make sure `/app/answers.json` is valid JSON with one entry per question id.
