# ft-2026.09 incident timeline

* day 0 - release ships; offline suite green, no slice below threshold
* day 2 - support escalates quality complaints concentrated in the watched slice
* day 3 - online dashboard confirms a drop; offline suite re-run on the shipped checkpoint, still green
* day 4 - rollback rejected: the knob the release changed cannot be fully reverted
* day 4 - three hypotheses on the table, no way to choose between them from the numbers we have:
  the student is too small for this slice / the mixture starved the slice / the recipe lr is wrong now
* day 5 - you are asked to find which one it is and ship a fix
