# Packer rewrite: behavioural details preserved from `pack_ref.py`

`solution.py` reproduces the reference bin ids exactly. It keeps the remaining capacities of
the *open* bins in a compact numpy buffer (in bin-creation order) so the first-fit scan and the
"fullest bin" search are single vectorised calls instead of Python loops. On the deployment
parameters a 400,000-sequence shard packs in about 0.5-1.2 s (reference: ~35-47 s).

## Reference semantics that had to be kept

1. **Bin ids are creation order.** Every new bin (regular or oversized) takes the next id;
   ids are never reused and closed bins keep their id.
2. **Oversized sequences (`L > C`)** get a bin of their own that is created *closed*: it never
   takes part in first-fit, never counts toward `max_open`, and is never the "oldest open bin".
   `L == C` is *not* oversized; it goes through normal placement (fits only a fresh bin).
3. **Oversized sequences skip the latency guard.** The reference `continue`s before the
   `(i + 1) % K` check, so if the sequence at a flush position is oversized, no flush happens
   that round. The counter `i` is the global sequence index, not a count of packed sequences.
4. **First fit in creation order** over the open bins only: the lowest-id open bin with
   `remaining >= L` wins. A sequence of length 0 (or negative) therefore always lands in the
   oldest open bin, or opens a new bin if none is open.
5. **Rule (1), close-below:** after placement, if `remaining < close_below` the bin is closed.
   Evaluated with the exact `<` (so `close_below = 0` never closes, and `close_below > C`
   closes every bin after its first sequence).
6. **Rule (2), memory guard:** evaluated *after* rule (1), with a strict `>`
   (`len(open) > max_open`). It removes the open bin with the smallest remaining capacity,
   ties broken by the smallest bin id. Because the buffer is kept in creation order, the
   first minimum (`argmin`) is exactly the smallest-id tie-break. Only one bin is removed per
   sequence, even if the invariant is violated by more (it cannot be, but the order matters).
   `max_open = 0` is legal and closes every bin immediately.
7. **Rule (3), latency guard:** evaluated last, only when `(i + 1) % K == 0` **and** at least
   two bins are open; it removes the oldest (lowest id) open bin. It runs after rules (1)
   and (2) so it sees their removals; e.g. if rule (1) closed the only other bin, nothing is
   flushed.
8. **Order of the three rules is (1) -> (2) -> (3)** on every packed sequence, and the count
   of open bins used by (2) and (3) reflects the removals made earlier in the same step.
9. **Only one placement per sequence:** a bin whose remaining capacity has dropped is never
   "reopened"; closed is permanent.
10. **Empty input** returns `[]`. Lengths equal to the capacity, zero lengths, negative
    close_below, `flush_every = 1`, etc. all follow from the rules above with no special cases.
11. **Value domain.** Lengths are treated as plain integers; numpy integer scalars are
    normalised to Python ints. Inputs that are not int64-safe (floats, huge ints) are routed
    through a pure-Python path with identical semantics, so results still match the reference.

## Validation performed

* `sample/sample_00.json` reproduced exactly.
* 7,000 randomised parameter/length combinations (capacities 0-4096, `close_below` from -5
  to C+1, `max_open` 0-1000, `flush_every` 1-1000, lengths including 0, C, C+1, negatives)
  compared element-for-element with `pack_ref.pack`.
* Three synthetic 400k shards (lognormal, uniform, large-only lengths) and cap-saturated
  adversarial inputs (all 2049, alternating 4000/3, near-tie remaining capacities) compared
  with the reference output: all identical.
