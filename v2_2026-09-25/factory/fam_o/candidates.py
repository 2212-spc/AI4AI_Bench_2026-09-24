"""Family O difficulty library: plausible-but-wrong packers.

Each candidate is the reference loop with exactly ONE semantic detail changed - the kind of detail a
strong solver drops while rewriting the O(n*open) scan into an O(n log n) data structure.  The generator
mines, for every candidate, a small input on which it disagrees with the reference; those inputs go into
the hidden test suite.  That makes "no shortcut" a mechanical, executable certificate instead of a claim.
"""


def _pack(lengths, params, v):
    C = params["capacity"]; t = params["close_below"]; m = params["max_open"]; K = params["flush_every"]
    remaining = []; open_ids = []; out = []
    for i, L in enumerate(lengths):
        if L > C:
            if v == "oversize_noid":           # forgets that an oversized item still consumes a bin id
                out.append(len(remaining) - 1 if remaining else 0)
                continue
            if v == "oversize_open":           # puts it in a normal, open bin
                remaining.append(C); out.append(len(remaining) - 1); open_ids.append(len(remaining) - 1)
                continue
            if v == "flush_oversize":          # runs the latency guard for oversized items too
                remaining.append(0); out.append(len(remaining) - 1)
                if (i + 1) % K == 0 and len(open_ids) >= 2:
                    open_ids.remove(open_ids[0])
                continue
            remaining.append(0); out.append(len(remaining) - 1); continue
        placed = -1
        if v == "bestfit":                     # best fit instead of first fit
            cands = [b for b in open_ids if remaining[b] >= L]
            if cands: placed = min(cands, key=lambda b: (remaining[b], b))
        elif v == "lastfit":
            for b in reversed(open_ids):
                if remaining[b] >= L: placed = b; break
        else:
            for b in open_ids:
                if remaining[b] >= L: placed = b; break
        if placed < 0:
            remaining.append(C); placed = len(remaining) - 1; open_ids.append(placed)
        remaining[placed] -= L; out.append(placed)

        def close_rule():
            if v == "close_le":
                if remaining[placed] <= t and placed in open_ids: open_ids.remove(placed)
            else:
                if remaining[placed] < t and placed in open_ids: open_ids.remove(placed)

        def cap_rule():
            if len(open_ids) > m:
                if v == "cap_emptiest":
                    x = max(open_ids, key=lambda b: (remaining[b], -b))
                elif v == "cap_tie_high":
                    x = min(open_ids, key=lambda b: (remaining[b], -b))
                elif v == "cap_oldest":
                    x = open_ids[0]
                else:
                    x = min(open_ids, key=lambda b: (remaining[b], b))
                open_ids.remove(x)

        def flush_rule():
            need = 1 if v == "flush_ge1" else 2
            if (i + 1) % K == 0 and len(open_ids) >= need:
                if v == "flush_newest": open_ids.remove(open_ids[-1])
                else: open_ids.remove(open_ids[0])

        if v == "cap_before_close":
            cap_rule(); close_rule(); flush_rule()
        elif v == "flush_before_cap":
            close_rule(); flush_rule(); cap_rule()
        elif v == "no_cap":
            close_rule(); flush_rule()
        elif v == "no_flush":
            close_rule(); cap_rule()
        else:
            close_rule(); cap_rule(); flush_rule()
    return out


VARIANTS = ["bestfit", "lastfit", "oversize_noid", "oversize_open", "flush_oversize", "close_le",
            "cap_emptiest", "cap_tie_high", "cap_oldest", "flush_before_cap",
            "no_cap", "no_flush", "flush_ge1", "flush_newest"]

# Provably equivalent to the reference: when the memory guard can fire, the just-placed bin is the unique
# open bin below the closure threshold, so closing-then-guarding and guarding-then-closing agree.  Kept as a
# positive control - the difficulty gate must NOT reject it, otherwise the gate is merely paranoid.
EQUIVALENT = ["cap_before_close"]


def candidate(v):
    return lambda lengths, params: _pack(lengths, params, v)
