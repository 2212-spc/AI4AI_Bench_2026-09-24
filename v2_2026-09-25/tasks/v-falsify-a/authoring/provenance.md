# Provenance - family V, seed 20260924

**Seed corpus.** Real data-infrastructure incidents: MinHash/LSH deduplication recall, n-gram
decontamination that normalises one side only, `pad_to_multiple` sharding that repeats the tail document,
block-local shuffling that drops the remainder.

**Deep transformation applied.** Bug-finding benchmarks (SWE-bench and its relatives) give the solver a
failing test and ask for a patch; the failure is handed over. Here nothing fails: the code runs, the
pipeline produces plausible output, and the *claims* are the artefact under test. The deliverable is a
constructed input, not a patch, and the solver must also refuse to attack the two claims that are true -
which turns the usual "find the bug" incentive into a two-sided decision. The V1 witness cannot be found
by testing at all: the banding parameters put random escape at 3e-12, so the solver has to reason
about what the signature does and engineer shingles against it.

**Truth source.** Self-evidencing. For a falsifiable claim there is no stored answer: the verifier runs
the witness and recomputes the claimed quantity independently, so the witness proves itself. For the two
claims recorded as holding, the key is backed by a 4000-input fuzz per claim and by grading asymmetry -
a confirmed witness is accepted even against the key, so a wrong key entry can never fail a correct agent.

**Counter-shortcut evidence.** `certificate.json` records: 1200 random near-duplicate pairs produce
0 V1 witnesses; five degenerate or out-of-scope witnesses are all rejected with reasons; the
blanket "everything is violated" and "everything holds" answers both fail; blind labelling succeeds with
probability 0.016 before witnesses are even considered.
