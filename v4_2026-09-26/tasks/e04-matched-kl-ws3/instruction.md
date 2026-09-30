# Task: Matched-KL ship review: pre-register the experiment that decides best-of-n against PPO

You are a research engineer in a post-training team.  The team's notebook (`/app/notebook/`) holds earlier runs and the team's notes.  You have a budget on the team's simulated post-training service (`lab` command; read `/app/manual.md` first - it defines the service, its guarantees and how answers are graded).

Answer every question below by writing `/app/answers.json` (a template with the right keys is already there).  You may run any experiments within the budget.  Work carefully: the notes may be wrong, and some questions may not be fully answerable from anything the lab can measure - the manual says how to answer those.

## Questions

- **q1** The team's proposal is `method=ppo kl=6.8405 steps=579`.  What best-of-n sampling budget `n` puts a policy at the **same** KL from the initial policy as that run actually reaches?  Answer the exact n, not a bound.  [unit: samples] Answer format: `{"lo": "number", "hi": "number"}`
- **q2** Of the proposed recipe's proxy reward - its gain over the initial policy, which scores 0 - what fraction is contributed by the length term rather than by quality?  Answer a fraction in [0, 1].  [unit: fraction of proxy reward] Answer format: `{"lo": "number", "hi": "number"}`
- **q3** The team will ship exactly one of the two recipes below, and they are to be compared at the
**same distance from the initial policy** - the KL the proposed PPO recipe actually reaches:

  - `bon`: best-of-n sampling, at whatever sampling budget n puts it at that same KL;
  - `ppo`: the proposed recipe, `method=ppo kl=6.8405 steps=579`.

The ship rule is fixed: the answer is `bon` if best-of-n's true gold win rate exceeds the PPO recipe's by
more than 0.1, `ppo` if the PPO recipe's exceeds best-of-n's by more than 0.1, and `undetermined`
otherwise.

Pre-register the experiment that decides it: at most 3 runs costing at most 4400 in total, and a
decision rule over their results.  Your plan is executed in **every world consistent with what this
notebook disclosed**, each under several independent noise salts, and it passes only if the rule returns
that world's correct conclusion every time.  A rule that is right only in the world you measured yourself
will not pass, and neither will a statistic that cannot tell those worlds apart.  Answer format: `{"runs": "list of lab requests, each optionally with a \"label\"", "rule": {"expr": "arithmetic over <label>.<field>", "cuts": "[[op, threshold, label], ..., [\"else\", label]]"}}`
    - Plan limits: at most 3 runs, 4.4e+03 in total; allowed conclusions: `bon`, `ppo`, `undetermined`.
- **q4** There is budget for exactly one more alignment run before the freeze.  Which of these four
recipes gives the highest **gold** win rate?

  - `ppo_ship`: `method=ppo kl=6.8405 steps=579`
  - `ppo_light`: `method=ppo kl=6.8405 steps=431`
  - `bon_a`: `method=bon n=48`
  - `bon_b`: `method=bon n=246 len_pen=7.9563`

The first three carry no length penalty.  Answer format: `{"choice": "one of ppo_ship, ppo_light, bon_a, bon_b"}`

When you are done, make sure `/app/answers.json` is valid JSON with one entry per question id.
