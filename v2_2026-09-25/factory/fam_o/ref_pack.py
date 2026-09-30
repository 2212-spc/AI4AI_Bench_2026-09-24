"""Reference sequence packer for MiniStack's training data loader.

This file is the specification: the packing produced by `pack()` is, by definition, correct.
It is shown to the agent (public).  Parameters are instance-specific and live in `params.json`.
"""


def pack(lengths, params):
    """Assign every sequence to a bin and return the bin id of each sequence.

    Bins are numbered 0, 1, 2, ... in the order they are created.

    capacity      C : token capacity of a bin
    close_below   t : after a sequence is placed, a bin whose remaining capacity is
                      strictly below t is closed (it can never receive another sequence)
    max_open      m : at most m bins may be open at once
    flush_every   K : latency guard, see below
    """
    C = params["capacity"]
    t = params["close_below"]
    m = params["max_open"]
    K = params["flush_every"]

    remaining = []   # remaining[b] = free capacity of bin b (creation order)
    open_ids = []    # ids of the currently open bins, in creation order
    out = []

    for i, L in enumerate(lengths):
        if L > C:
            # A sequence longer than the capacity is truncated by the tokenizer upstream and
            # shipped in a bin of its own, which is never open for further packing.
            remaining.append(0)
            out.append(len(remaining) - 1)
            continue

        placed = -1
        for b in open_ids:                      # first fit, in bin creation order
            if remaining[b] >= L:
                placed = b
                break
        if placed < 0:
            remaining.append(C)
            placed = len(remaining) - 1
            open_ids.append(placed)

        remaining[placed] -= L
        out.append(placed)

        if remaining[placed] < t:               # (1) too little room left to be useful
            open_ids.remove(placed)

        if len(open_ids) > m:                   # (2) memory guard: flush the fullest open bin
            v = min(open_ids, key=lambda b: (remaining[b], b))
            open_ids.remove(v)

        if (i + 1) % K == 0 and len(open_ids) >= 2:
            # (3) latency guard: every K packed sequences the oldest open bin is flushed so that
            #     no bin can sit in the buffer forever.
            open_ids.remove(open_ids[0])

    return out
