# Stack and traffic model

Latency of one request is the sum of the time it spends in each of the stack's internal stages.  Two facts
about the queued changes, from the vendor's engineering notes:

* Each change **divides** the stage or stages it touches by a fixed factor, and **adds** a small fixed
  fraction of one other stage.  Those factors and fractions are properties of the change.  They do not
  depend on which request family is running, and they do not depend on which other changes are shipped
  alongside it.
* How much of a request's time lands in each stage depends on the **request family**.  The stage breakdown
  is not exported by the runtime, for any family.

Consequences worth being explicit about, because the last two quarterly plans were rejected over them:

* A change's measured effect on one request family does not carry over to another family, and the effect of
  a *set* of changes is not the combination of their individual effects - two changes that relieve the same
  stage overlap, and a change that looks worthless alone can be the best one once another change has
  removed the stage that was dominating.
* Production is not one of the replayable families.  It is traffic made of all five of them, in proportions
  nobody has measured.

## The canary

The canary rollout gate exists because a bad change on live traffic costs real money, and it admits one
queued change per window.  This is a hard operational limit, not a budget matter: there is no way to put a
multi-change bundle in front of production traffic before the plan is approved.
