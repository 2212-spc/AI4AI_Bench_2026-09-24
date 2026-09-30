"""Self-check for `d-design-open`: the shipped verifier rewards the reference and nothing else.

The open arm grades the same five shapes of answer as the sealed one and ships the same verifier, so the
probe set is the sealed arm's, re-run against this task's key.  What differs is the population of
*competing readings*: `gen_do` adds the two other fillings of the cell the recovery log never exercises,
and those are the probes that matter here.  Both are consistent with every row of the log; both are what
an agent that completes the table by elimination will hand in; both must score 0.  The wrapper therefore
imports `gen_do` before the sealed self-check, so `RECOV_RULES` already carries them when the impostor
sweep is built, and then checks that they really were probed rather than silently skipped.

Run as `python3 selfcheck_do.py [task_dir]`.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAM = os.path.join(os.path.dirname(HERE), "fam_d")
TASK = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench2/tasks/d-design-open"
sys.path.insert(0, FAM)
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0], TASK]

import gen_do as GD                                                           # noqa: E402
import gen_ds as GS                                                           # noqa: E402

assert GS.RIVAL in GS.RECOV_RULES and GS.RIVAL != GS.NAIVE_RULE, GS.RIVAL
assert all(n in GS.RECOV_RULES for n in GD.COMPLETIONS), sorted(GD.COMPLETIONS)

import selfcheck_ds as SC                                                     # noqa: E402

assert SC.TASK == TASK, (SC.TASK, TASK)
rc = SC.main()
missing = [n for n in GD.COMPLETIONS if n != GS.TRUTH_RULE]
print("probed completions:", missing)
sys.exit(rc)
