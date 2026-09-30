# RLLab manual

## 1. What the service models (abstraction boundary)

RLLab is a **simulated** post-training service.  No real model is trained; every request is answered by
a simulator.  The simulator is not a replica of any real RLHF stack.  Its mechanisms are modelled on
regularities reported in the alignment literature, but **its constants were drawn fresh for this lab**.
Published numbers (KL budgets, reward-model accuracies, best-of-n sweet spots) describe other worlds.
Measure; do not recall.

A **policy** is identified by its training configuration.  Re-stating the same training knobs with
`svc=gold` evaluates *that* policy; there is no separate policy id and nothing is cached.

The simulator contains the following kinds of effects.  **Any given lab may switch some of them off
(held neutral); which ones are active is not stated.**

- *Proxy versus gold.*  The reward model's score and the true (human) quality are different functions of how far the policy has moved from its initialisation.  Distance is measured as d = sqrt(KL).
- *Reachable distance.*  Best-of-n sampling and policy-gradient training reach a given distance in different ways: one has a closed-form relation between n and KL, the other approaches a target KL over steps.
- *Reward-model data.*  A reward model's held-out accuracy grows with the number of preference comparisons it was trained on, up to an asymptote set by its size, and is at chance below a data floor.
- *Reward-model quality.*  How fast the proxy and the gold diverge depends on the reward model.
- *Length.*  Mean answer length can grow as the policy moves, and part of the proxy reward can be a function of length alone.  A length penalty can hold length back.
- *Preference-label noise.*  Noise in the preference labels affects the reward model's accuracy and its calibration; the two need not be affected to the same degree.
- *Entropy.*  Policy entropy falls as the policy moves; reward and entropy may be tied by a fixed relation with a ceiling.
- *Compute scaling.*  Performance against training compute may follow a saturating curve; recipe choices may move where the curve rises without moving what it saturates at.
- *Noise.*  Proxy reward, length and entropy carry run-to-run noise; a gold evaluation with `gold_n`
  comparisons carries binomial noise.

Not modelled: tokenizer, wall-clock time, specific optimizers, prompt distribution shift.

## 2. Guarantees

- **Fixed laws.**  Every mechanism is a fixed law whose constants do not depend on which request you
  make.  There are no hidden thresholds, **except** where Section 5 lists a *known unknown*.
- **Determinism.**  An identical request with the same seed returns identical numbers (and is charged
  again).  Different seeds give independent draws.  `kl` is reported without noise.
- **Notebook rows are real service responses.**  The team's notes are the team's interpretation and may be wrong.

## 3. How answers are graded

Every question is graded mechanically against this lab's hidden world.  Nothing you write outside the
answer file is read, and there is no credit for an explanation of a wrong answer.

- **Numeric questions** take an interval `{"lo": x, "hi": y}`.  The interval must be the *set of values
  consistent with everything that can be known*: what the lab can measure within your budget, plus the
  guarantees and documented ranges in this manual.
  - If the lab pins a quantity down, answer a degenerate interval (`lo == hi` = your estimate).  Seed
    noise is handled by the grading tolerance, so **do not** widen for statistical uncertainty.
  - If the quantity depends on a known unknown (Section 5), `lo`/`hi` are the minimum and maximum over
    the values of that unknown that are consistent with its documented range **and** with what the
    lab can measure.
  - Each endpoint is compared with the true endpoint.  The tolerance is calibrated to how precisely a
    well-designed set of experiments within the budget can pin that endpoint down.
- **Decisions** take `{"choice": "<option>"}`: pick the option with the best outcome as defined in the question.
- **Plan questions** ask for the experiment you *would* run and the rule you would apply to its result,
  and are graded on the **plan**, not on a number.  The answer for such a question is the plan object
  itself:

        {"runs": [{"label": "r1", ...a lab request...}, ...],
         "rule": {"expr": "r1.acc - r2.acc",
                  "cuts": [[">", 0.05, "A"], ["<", -0.05, "B"], ["else", "undetermined"]]}}

  - Each run is an ordinary lab request (the same fields `lab run` takes).  `label` is a name you choose
    for referring to that run's output in the rule; it is not a knob.  The plan must stay within the run
    count and budget the question states - that budget is separate from your own exploration budget, and
    a plan that exceeds it fails without being executed.
  - `expr` is arithmetic over `<label>.<field>`, where `<field>` is any numeric field of that run's
    result.  Repetition accuracies are `<label>.acc_r0`, `acc_r1`, ...; per-slice fields are
    `<label>.<field>_s0`, `_s1`, ...; per-order fields are `<label>.<field>_<order>`.  Allowed:
    `+ - * / **`, parentheses, numeric constants, and `abs min max sqrt log exp mean`.  Nothing else
    parses - no attribute chains, no comprehensions, no calls to anything unlisted.
  - `cuts` are tried in order and the first match wins; each is `[op, threshold, label]` with `op` one of
    `> >= < <=`.  The last cut must be `["else", label]`: the rule has to name a conclusion for every
    value the statistic could take.  Labels must come from the set the question lists.
  - **Grading.**  The plan is executed by the grader in *every world consistent with what was disclosed*
    - the true one and others that fit the notebook equally well - each under several independent noise
    salts.  It passes only if the rule returns that world's correct conclusion **every** time.  A rule
    that is right in the world you happen to be in and flips in another consistent world fails; so does
    one whose runs do not buy enough precision to separate the cases at the budget given.  Size the runs
    from the noise model in Section 1 before choosing the thresholds.

## 4. Using the service

    lab spec
    lab run svc=rm_eval rm_size=1b rm_data=20000 seed=0
    lab run svc=train method=bon n=64 rm_size=1b rm_data=20000 len_pen=0 seed=0
    lab run svc=train method=ppo kl=6 steps=800 rm_size=1b rm_data=20000 len_pen=0 seed=0
    lab run svc=gold method=ppo kl=6 steps=800 rm_size=1b rm_data=20000 len_pen=0 gold_n=200 seed=0
    lab batch plan.json
    lab status
    lab history

`train` reports `kl`, `proxy_reward`, `mean_len`, `entropy`.  `gold` reports `gold_wins` out of
`gold_n` comparisons against the reference policy.  `rm_eval` reports `rm_acc` and `rm_ece` on a fixed
held-out preference set.  A gold evaluation is far more expensive per unit of information than a
training run: budget for it.

## 5. This lab

Settable knobs:
- `svc`: one of train, gold (default train)
- `method`: one of bon, ppo (default ppo)
- `kl`: number in [0.05, 12] (default 1.0)
- `steps`: number in [10, 1000] (default 400)
- `n`: number in [1, 4096] (default 1)
- `len_pen`: number in [0, 8] (default 0.0)
- `gold_n`: number in [50, 2e+04] (default 200)

Fixed settings (cannot be changed here): `rm_size`=rm_v3, `rm_data`=4e+04

Budget: at most 7e+03 GPU-hours per request, 9e+03 GPU-hours in total, at most 130 requests.  Charging: train: steps x size factor; gold: gold_n x human factor; rm_eval: flat.

This deployment has **one** reward model, `rm_v3`, frozen before any of the runs in the notebook. `rm_size` and `rm_data` are not settable and `svc=rm_eval` is not available in this build; the model's held-out preference accuracy is 0.7144 and its calibration is not in question here.  How fast the proxy and the gold diverge as the policy moves away from its initialisation is a property of *this* reward model and is not disclosed anywhere.

`svc=train` reports `kl`, `proxy_reward`, `mean_len` and `entropy`.  `svc=gold` reports `gold_wins` out of `gold_n` independent human comparisons against the team's reference policy, and its `gold_win_rate` is the only measurement in this lab that does not come from the reward model.  `entropy` is reported for diagnostics; no question here depends on it.

`len_pen` accepts values in [0, 8] and holds mean length back at a fixed training target - it does not change how far the policy moves, and a `train` row and a `gold` row with identical training knobs describe the same policy.  `n` accepts up to 4096, which is a limit of this lab and not of the serving stack.

`steps` accepts values in [10, 1000].  That is a limit of this deployment's scheduler, not of the training stack: the team's own plan runs longer than this lab will run, and a request above the limit is rejected rather than truncated.

### Known unknowns

- None: everything the questions ask about is determined by the lab plus Section 2.
