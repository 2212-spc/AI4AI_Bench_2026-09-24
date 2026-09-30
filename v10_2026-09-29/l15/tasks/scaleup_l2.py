"""SCALEUP level L2 ("scaleup_dup") = L1 + one data operator, SAME task text (minimal pair).

Operator D1 - DUPLICATED CORPUS.  The 80,000-row corpus is 20,000 distinct labelled rows, each present 4 times
(exact copies: same features, same crowd label), in random order.  Every sentence of the instruction stays true
(80,000 rows; the sample is a 5% uniform sample of them; production makes the same 32 passes over the rows).
What changes is the regime production trains in: 20,000 distinct rows seen ~128 times each, not 80,000 rows
seen 32 times.

Why this is the S7 step for this family (the L1 frontier solutions become the decoys):
  * L1's reference - scale AdamW's wd by N_sample / N_full at equal epochs - now under-regularises badly
    (seed 0: 0.616 vs 0.723 for shipping the team config unchanged).  The row count is the wrong N.
  * the L1-validated experiment - subsample the sample at equal passes, watch wd* move, extrapolate to 80,000
    rows - measures the wrong regime (few passes over distinct rows) and extrapolates to the wrong N.
  * in-job tuning on a random held-out slice of the corpus (Fable's L1 solution, shipped here verbatim as a
    decoy) now validates on copies of training rows: memorisation looks like accuracy, the smallest wd wins
    (seed 0: 0.624).
What is still identifiable from what the agent has: 7% of the sample's rows are exact duplicates of another
sample row (birthday count; the 4,000-row sample holds ~290 duplicate pairs, so k_hat = 1 + pairs (N-1)/C(n,2)
~ 3.9); the production regime (N/k distinct rows, 32k passes) can be replicated at sample scale on the
sample's distinct rows, or tuned in-job on a held-out slice of DISTINCT rows.
What L2 does NOT defeat (measured, reported, not hidden): a noise-robust loss (the crowd noise is uniform, so a
forward-corrected likelihood is the right model) with a hedged decay rule - GPT-6's L1 solution - clears the
L2 bar on seed 0 (0.791).  L2 is a hardening against the scaling-experiment and in-job-holdout habits, not
against every L1 solution.

Everything else (grader, sandbox, CPU cap, instruction text, starter files) is inherited from scaleup.py."""
from .scaleup import *            # noqa: F401,F403
from . import scaleup as _b

LEVEL = "L2"
DUP = 4
CERT_PATH = _b.CERT_PATH_L2


def base_params(seed):
    p = _b.base_params(seed)
    p["dup"] = DUP
    return p


def sample_params(seed):
    p = base_params(seed)
    c = _b._cert_all(DUP).get(str(seed))
    if not c or "T" not in c:
        raise RuntimeError("scaleup_l2 seed %d is not certified; run python3 -m l15.tasks.scaleup_l2_cert" % seed)
    p.update({"team_lr": c["team"]["lr"], "team_wd": c["team"]["weight_decay"], "T": c["T"],
              "job_cpu_s": c["job_cpu_s"]})
    return p


class World(_b.World):
    NAME = "scaleup_l2"
