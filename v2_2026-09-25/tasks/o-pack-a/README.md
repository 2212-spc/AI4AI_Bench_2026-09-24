# Family O instance (seed 20260924) - exact-equivalence acceleration of a spec-as-code packer

Not shown to the agent.

## Difficulty
The agent must turn an O(n * open_bins) first-fit loop into an O(n log n) structure while preserving nine
interacting behaviours, three of which are invisible on ordinary inputs (oversized sequences still consume
a bin id; the latency guard is driven by the input index, so oversized sequences advance it but never
trigger it; the memory guard closes the *fullest* bin with ties broken by smallest id). A leftmost-index
query with deletions needs a segment tree, and the memory guard needs a second order on the same set, so
the obvious "heap of open bins" answer is wrong twice over. 7 single-rule variants of the reference
were each falsified by a mined input that is part of the hidden suite, so dropping any one rule scores 0.
Timing: specification implementation 47.39s vs blind reference 1.68s on the
400k shard (ratio 28.2x); the limit is 12.0s at calibration factor 1.

## Reference solution
`solution/solution.py`: two segment trees over bin creation index (max-remaining for the leftmost-fit
descent, min-remaining for the memory guard) plus a monotone pointer for the oldest open bin. Written from
the specification only.

## Verification
`tests/verify_o.py` imports `/app/solution.py` in a fresh process and compares its output element for
element against `tests/expected.npz`, which was produced by executing `pack_ref.py` at authoring time on
40 hidden inputs (8 length profiles x 3 seeds, 7 mined corner-case inputs, 6 hand-built edge
inputs) plus one 400k timing shard. Reward 1 iff every case matches and the timing shard is packed inside
the calibrated limit. No rubric, no statistics, no judgement.
