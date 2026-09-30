"""Removal of evaluation documents that overlap the training corpus by a long n-gram."""
NGRAM = 13
_PUNCT = str.maketrans({c: " " for c in ",.;:!?\"'()[]{}<>/\\|`~@#$%^&*_+=-"})


def _normalise(text):
    return text.lower().translate(_PUNCT)


def build_index(train_docs):
    idx = set()
    for d in train_docs:
        w = _normalise(d).split()
        for i in range(len(w) - NGRAM + 1):
            idx.add(" ".join(w[i:i + NGRAM]))
    return idx


def filter_eval(eval_docs, index):
    """Surviving evaluation documents, in input order."""
    out = []
    for d in eval_docs:
        w = d.split()
        hit = any(" ".join(w[i:i + NGRAM]) in index for i in range(len(w) - NGRAM + 1))
        if not hit:
            out.append(d)
    return out
