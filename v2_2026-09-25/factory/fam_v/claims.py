"""Family V claims, their scope, and violation predicates computed independently of the shipped code.

Each predicate re-derives the quantity the claim talks about from the witness itself (its own shingling,
its own n-gram extraction, its own assignment counting) and only then calls the module under test.  A
"violated" verdict is therefore established by a check that does not trust the module.

The modules used here are imported from the PRISTINE copy that ships with the tests, never from /app.
"""
import os, sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "dataprep"))
import dedup, decontam, shard, sampler, pack, shuffle_buffer                       # noqa: E402

CLAIM_TEXT = {
    "V1": ("dedup.dedup: for any two documents of at least 300 and at most 4000 whitespace tokens whose "
           "5-token-window sets have Jaccard similarity at least 0.80, dedup never keeps both."),
    "V2": ("decontam.filter_eval: no evaluation document returned by filter_eval shares a 13-token window "
           "(whitespace tokens, exact string match, no normalisation) with any training document."),
    "V3": ("shard.shard: for any n in [1, 10^6] and world_size in [1, 16] with drop_last=False, the "
           "assignments over all ranks contain every index in [0, n) exactly once."),
    "V4": ("pack.pack: no emitted batch has a total token count above max_tokens, where a document longer "
           "than max_tokens counts as max_tokens."),
    "V5": ("shuffle_buffer.shuffle_stream: the emitted sequence is a permutation of the input items for "
           "any buffer_size >= 1."),
    "V6": ("sampler.epoch_order: for any n in [1, 10^6], any seed and any epoch, the returned order "
           "contains every index in [0, n) exactly once."),
}
SCHEMA = {
    "V1": '{"docs": [str, str]}  - two documents, 300..4000 whitespace tokens each, printable ASCII',
    "V2": '{"train": [str, ...], "eval": [str, ...]}  - 1..20 documents per side, <=2000 tokens each',
    "V3": '{"n": int, "world_size": int, "drop_last": false}  - 1<=n<=10^6, 1<=world_size<=16',
    "V4": '{"lengths": [int, ...], "max_tokens": int}  - <=5000 lengths, each 0..10^6, max_tokens>=1',
    "V5": '{"items": [int, ...], "buffer_size": int, "seed": int}  - <=5000 items, buffer_size>=1',
    "V6": '{"n": int, "seed": int, "epoch": int}  - 1<=n<=10^6, 0<=seed<2^31, 0<=epoch<=1000',
}


def _tok(s):
    return s.split()


def _wins(s, k):
    w = _tok(s)
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


def _need(cond, msg):
    if not cond:
        raise ValueError(msg)


def v1(wit):
    docs = wit["docs"]
    _need(isinstance(docs, list) and len(docs) == 2, "docs must be a list of two strings")
    for d in docs:
        _need(isinstance(d, str), "documents must be strings")
        n = len(_tok(d))
        _need(300 <= n <= 4000, "each document needs 300..4000 whitespace tokens, got %d" % n)
        _need(all(32 <= ord(c) <= 126 or c in "\n\t" for c in d), "documents must be printable ASCII")
    a, b = (_wins(d, 5) for d in docs)
    j = len(a & b) / len(a | b)
    _need(j >= 0.80, "jaccard of the 5-token-window sets is %.4f, below 0.80" % j)
    kept = dedup.dedup(docs)
    return len(kept) == 2, {"jaccard": round(j, 4), "kept": kept}


def v2(wit):
    tr, ev = wit["train"], wit["eval"]
    for side, name in ((tr, "train"), (ev, "eval")):
        _need(isinstance(side, list) and 1 <= len(side) <= 20, "%s must hold 1..20 documents" % name)
        for d in side:
            _need(isinstance(d, str) and len(_tok(d)) <= 2000, "%s documents must be <=2000 tokens" % name)
    train_ng = set()
    for d in tr:
        train_ng |= _wins(d, 13)
    surv = decontam.filter_eval(ev, decontam.build_index(tr))
    leaked = [i for i, d in enumerate(surv) if _wins(d, 13) & train_ng]
    return bool(leaked), {"n_surviving": len(surv), "leaking_survivors": leaked[:5]}


def v3(wit):
    n, ws = int(wit["n"]), int(wit["world_size"])
    _need(1 <= n <= 10 ** 6, "n must be in [1, 10^6]")
    _need(1 <= ws <= 16, "world_size must be in [1, 16]")
    _need(not wit.get("drop_last", False), "the claim is scoped to drop_last=False")
    c = Counter()
    for r in range(ws):
        c.update(shard.shard(n, ws, r, drop_last=False))
    bad = [i for i in range(n) if c[i] != 1]
    return bool(bad) or bool(set(c) - set(range(n))), {"n_indices_not_once": len(bad),
                                                      "example": bad[:5],
                                                      "out_of_range": sorted(set(c) - set(range(n)))[:5]}


def v4(wit):
    L, mt = [int(x) for x in wit["lengths"]], int(wit["max_tokens"])
    _need(1 <= len(L) <= 5000, "lengths must hold 1..5000 entries")
    _need(all(0 <= x <= 10 ** 6 for x in L), "each length must be in [0, 10^6]")
    _need(mt >= 1, "max_tokens must be >= 1")
    over = [b for b in pack.pack(L, mt) if sum(min(L[i], mt) for i in b) > mt]
    return bool(over), {"n_batches_over": len(over), "example": over[:3]}


def v5(wit):
    items, bs = [int(x) for x in wit["items"]], int(wit["buffer_size"])
    _need(1 <= len(items) <= 5000, "items must hold 1..5000 entries")
    _need(bs >= 1, "buffer_size must be >= 1")
    out = shuffle_buffer.shuffle_stream(list(items), bs, int(wit.get("seed", 0)))
    return Counter(out) != Counter(items), {"n_in": len(items), "n_out": len(out)}


def v6(wit):
    n, seed, ep = int(wit["n"]), int(wit["seed"]), int(wit["epoch"])
    _need(1 <= n <= 10 ** 6, "n must be in [1, 10^6]")
    _need(0 <= seed < 2 ** 31 and 0 <= ep <= 1000, "seed/epoch out of scope")
    order = sampler.epoch_order(n, seed, ep)
    c = Counter(order)
    bad = [i for i in range(n) if c[i] != 1]
    return bool(bad) or bool(set(c) - set(range(n))), {"n_returned": len(order),
                                                      "n_indices_not_once": len(bad),
                                                      "example": bad[:5]}


CHECK = {"V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5, "V6": v6}
