# Provenance - family C, seed 20260924

**Seed corpus.** The ablation-table culture of systems ML papers (a knob, a baseline, a delta) and the
observational-causal-inference benchmarks that score effect estimates against a known structural model.

**Deep transformation applied.** Published causal benchmarks hand the solver a graph, or at least a fixed
covariate set, and ask for one effect; ablation tables hand over a randomised design. Here the graph is
withheld, the covariate set has to be chosen from *when* each column was measured, the design is the
residue of human scheduling rather than an experiment, and - the part no existing benchmark scores - the
solver has to return "this log cannot answer that" for the right reason on some queries while refusing to
over-abstain on the rest. The deliverable is a verdict per query plus a number, not a single number.

**Truth source.** Analytic: the interventional contrast is evaluated in closed form on the structural
model in `world.py`, which is *not* shipped to the agent. The hardware term and all noise are additive, so
they cancel in the contrast and truth is exact, not estimated.

**Independence.** Truth (closed form from the structural equations) and the reference answer (a blind
least-squares analysis of the exported csv) are computed by disjoint code paths; the certificate records
their agreement.

**Counter-shortcut evidence.** `certificate.json` lists 10 flawed procedures - additive-only,
probe-omitted, during-run-adjusted, marginal means, estimability-check-removed, numeric encodings,
throughput-as-hardware-proxy, and the two degenerate all-identified / all-underdetermined answers - each
with the query that catches it, plus 1 alternative-but-valid procedure(s) the gates must not reject.
