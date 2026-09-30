# Falsify the data team's promises

`/app/dataprep/` is MiniStack's data preparation stack: near-duplicate removal, decontamination,
sharding, packing, streaming shuffle, and the per-epoch sampler. `/app/CLAIMS.md` lists six claims the
design doc makes about these modules, each with the scope it is asserted over and the schema of a witness
against it.

Some of the claims are true of the code as shipped. Some are false. Your job is to say which, and to
*prove* the false ones with a concrete input.

Beware: at least one of the false claims fails only on inputs that must be constructed deliberately -
sampling inputs at random and hoping to trip it will not work, and neither will assuming that a claim
about a probabilistic algorithm must be false just because the algorithm is probabilistic.

## Deliverable

`/app/findings.json`, one entry per claim id:

```json
{
  "V3": {"status": "violated", "witness": "witnesses/v3.json"},
  "V4": {"status": "holds",    "argument": "why no input in scope can violate it, >= 80 characters"}
}
```

Write each witness as a JSON file somewhere under `/app` and give its path relative to `/app`
(at most 2 MiB each).

## How it is graded

* A `violated` answer counts only if the witness parses, lies inside the claim's published scope, and an
  independent check - one that recomputes the claimed quantity from the witness itself rather than
  trusting the module - confirms that the module breaks the claim on it.
* A `holds` answer counts only if the claim really is unfalsifiable within its scope. The argument text is
  required but is not itself graded.
* A confirmed witness is always accepted, including for a claim the grader expected to hold. Finding
  something the authors missed cannot cost you anything.
* The grader imports its own pristine copy of `dataprep`. Editing `/app/dataprep` changes nothing; you are
  free to copy, instrument and experiment with it.
* Grading is all-or-nothing over the six claims.

Python 3 with numpy is available and there is no network.


You have 5400 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
