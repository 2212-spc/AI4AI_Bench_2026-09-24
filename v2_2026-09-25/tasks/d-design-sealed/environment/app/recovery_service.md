# Cold-storage recovery requests

The evaluation fleet writes a per-run **trace bundle** to cold storage.  A **recovery request** names a
set of launched runs and is answered out of those bundles.  Requests are slow and are charged to the
team's quarterly budget, which is why the planning questions in `queries.json` ask what a request would be
worth *before* it is issued.

Three things about the service matter for those questions:

  * a request is issued blind.  You commit to the whole set and pay for it; you do not get to see the
    first answer before choosing the second.
  * what comes back for a run is never wrong and never contradicts this archive, but it is **not the same
    for every run**.  Cold storage is a different system from the archive - written by the evaluation
    fleet, not by the archiver - and the incidents in `incidents.md` did not all reach the two systems the
    same way.  Nothing here, and nothing anywhere else in `/app`, states which runs answer which way:
    establishing that is part of the work.
  * the only record of how the service behaves is `recovery_log.csv`: every recovery request the on-call
    team issued during the incident response, with what came back.  Each row gives the run, the outcome,
    and, where anything came back, the range of `pass_rate` that answer left the run in (`returned_lo`
    and `returned_hi`, equal to each other when the answer was a single number).  It is a sample - most
    launched runs were never requested, and none of the runs `queries.json` asks about were.  It was not
    collected to answer your questions and nothing in it is arranged for your convenience.  What it shows
    about the service holds for the whole fleet, including runs it does not mention.

Where an answer leaves a run in a range rather than at a number, that range is a genuine constraint on the
run's `pass_rate` and nothing more: the service does not promise the value sits anywhere in particular
inside it, and where the range falls is not something this archive determines.
