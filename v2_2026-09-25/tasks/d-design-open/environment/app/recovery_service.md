# Cold-storage recovery requests

The evaluation fleet writes a per-run **trace bundle** to cold storage.  A **recovery request** names a
set of launched runs and is answered out of those bundles.  Requests are slow and are charged to the
team's quarterly budget, which is why the planning questions in `queries.json` ask what a request would be
worth *before* it is issued.

A request is issued blind: you commit to the whole set and pay for it, and you do not get to see the first
answer before choosing the second.

## What a request can come back with

For each run in the set, a request comes back in exactly one of these three ways, and in no other:

  1. **the bundle is intact** - you get that run's `pass_rate`;
  2. **the bundle was redacted** - the trace is gone, and what survives is the quality screen's audit
     line, which kept the score only to the resolution its suite is bucketed at.  You get a range of that
     width containing the value.  The service does not promise the value sits anywhere in particular
     inside the range, and where the range falls is not something this archive determines;
  3. **there is no bundle** - nothing comes back for that run.

What comes back is never wrong and never contradicts this archive.

## What it depends on

Cold storage does not decide case by case.  Which of the three you get for a run depends only on **what
became of that run's result record**: whether the archive still has its number and, if not, which of the
incidents in `incidents.md` is the reason it does not.  Two runs the incidents left in the same state come
back the same way.  It does not depend on the worker, on when the run finished, on when you ask, on what
else is in the same request, or on the run's knob settings - except that the *width* of a redacted answer
follows the run's own suite, since the suite is what sets the audit resolution.

Which state comes back which way is not written down anywhere.  Cold storage is a different system from
the archive - written by the evaluation fleet, not by the archiver - and the incidents did not all reach
the two systems the same way.  Nothing here, and nothing anywhere else in `/app`, says which is which.

## The log

`recovery_log.csv` is the record of the recovery requests the on-call team issued during the incident
response: the run, the outcome, and, where anything came back, the range of `pass_rate` that answer left
the run in (`returned_lo` and `returned_hi`, equal to each other when the answer was a single number).

It is what it is - a sweep made under time pressure while some of the incidents were still open, on the
runs somebody happened to need at the time.  It is a sample, not a survey.  Most launched runs were never
requested, none of the runs `queries.json` asks about were, it was not collected to answer your questions,
and nothing arranged for it to cover every state the fleet's runs ended up in.
