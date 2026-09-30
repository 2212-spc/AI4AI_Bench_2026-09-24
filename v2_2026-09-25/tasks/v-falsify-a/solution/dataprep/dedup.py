"""Near-duplicate removal for the pretraining corpus (MinHash + LSH banding).

Shingles are the 5-token sliding windows of the whitespace-tokenised text.  A document is dropped when it
lands in an already-occupied LSH bucket, so the first member of each near-duplicate group survives.
"""
from hashing import P, h64, perms

K_SHINGLE = 5
BANDS = 16
ROWS = 2
N_PERM = BANDS * ROWS
_PERMS = perms(N_PERM)


def shingles(text):
    w = text.split()
    return {" ".join(w[i:i + K_SHINGLE]) for i in range(max(0, len(w) - K_SHINGLE + 1))}


def signature(text):
    xs = [h64(s) for s in shingles(text)]
    if not xs:
        return None
    return [min((a * x + c) % P for x in xs) for a, c in _PERMS]


def bands(sig):
    return [(b, tuple(sig[b * ROWS:(b + 1) * ROWS])) for b in range(BANDS)]


def dedup(docs):
    """Indices of the documents that survive deduplication, in input order."""
    seen, kept = {}, []
    for i, d in enumerate(docs):
        sig = signature(d)
        if sig is None:
            kept.append(i)
            continue
        keys = bands(sig)
        if any(k in seen for k in keys):
            continue
        for k in keys:
            seen[k] = i
        kept.append(i)
    return kept
