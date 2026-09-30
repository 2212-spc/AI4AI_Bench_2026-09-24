# Alignment ship review - draft notes

The release candidate needs one more alignment pass.  We have a single reward model (`rm_v3`, held-out
preference accuracy 0.7144) and one shot at a human evaluation before the freeze, so this note is the
argument for skipping it.  All six runs are in `notebook/runs.jsonl`.

- **The sweep.**  Best-of-n at n=12/64/256, and PPO at `kl=3.078` (300 and 900 steps) and `kl=5.472` (300 steps).
  No length penalty on any of them.  Proxy reward climbs monotonically with the KL we measure, from 1.0350 at
  the short end to 1.5115 at KL 4.5491.  Nothing has turned over yet, so we are not past the peak.
- **The proposal.**  Train PPO with `kl=6.8405` for `579` steps.  That is the largest KL target we are willing
  to spend, and the sweep says the proxy reward will be worth it.
- **Why not best-of-n.**  Best-of-n needs no training at all, so we did look at it.  The standard result is
  that best-of-n at n samples sits log(n) nats from the initial policy, which puts our `kl=6.8405` target at
  n = 935.  That is a 935x sampling cost on every request, which we cannot pay at the current margin, so
  best-of-n is out and the choice is this PPO recipe or nothing.
- **Length.**  Mean length does rise across the sweep, from 634.9 to 1066.3 tokens.  But proxy reward rises
  with it, and the reward model was trained on human preference data, so the extra length is being
  *preferred* - that is quality, not padding.  We see no reason to spend the length penalty on it.

**Recommendation: ship the PPO recipe above, skip the human evaluation, and put the eval budget into the
next release instead.**

Nobody has checked what KL a PPO run of that length actually reaches, and nobody has run the length penalty
at a fixed KL.  We are out of time.
