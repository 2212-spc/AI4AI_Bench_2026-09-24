"""Greedy token packing into batches of at most max_tokens tokens."""


def pack(lengths, max_tokens):
    """List of batches; each batch is a list of document indices."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    out, cur, tot = [], [], 0
    for i, L in enumerate(lengths):
        L = min(int(L), max_tokens)
        if cur and tot + L > max_tokens:
            out.append(cur)
            cur, tot = [], 0
        cur.append(i)
        tot += L
    if cur:
        out.append(cur)
    return out
