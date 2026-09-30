"""Reference witnesses for family V - existence proofs for the falsifiable claims.

The interesting one is V1.  The LSH banding is 16 bands x 2 rows, so a *random* pair at Jaccard 0.9 is
missed with probability (1 - 0.9**2)**16 ~ 1e-11: searching for a witness is hopeless.  A witness has to be
engineered.  It is enough to make the two signatures differ in at least one row of every band, and one row
of one band can be broken by appending to document A a single 5-token window whose value under that row's
permutation is below A's shared minimum.  Sixteen appended windows - one per band - therefore break all
sixteen bands while keeping the Jaccard similarity above 0.9.
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "dataprep"))
import dedup                                                                      # noqa: E402
from hashing import P, h64                                                        # noqa: E402

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma "
         "tau upsilon phi chi psi omega north south east west river stone forest valley harbour meadow "
         "lantern compass anchor quarry timber orchard cellar bridge tunnel market cobble thistle").split()


def v1_witness(seed=11, n_words=820):
    rng = random.Random(seed)
    base = [rng.choice(WORDS) for _ in range(n_words)]
    doc_b = " ".join(base)
    shb = dedup.shingles(doc_b)
    xs = [h64(s) for s in shb]
    mins = [min((a * x + c) % P for x in xs) for a, c in dedup._PERMS]
    phrases, tries = [], 0
    for band in range(dedup.BANDS):
        j = band * dedup.ROWS                          # break row j, hence band `band`
        a, c = dedup._PERMS[j]
        while True:
            tries += 1
            p = " ".join(["z%d" % tries] + [rng.choice(WORDS) for _ in range(4)])
            if (a * h64(p) + c) % P < mins[j]:
                phrases.append(p)
                break
    doc_a = doc_b + " " + " ".join(phrases)
    sha = dedup.shingles(doc_a)
    j = len(sha & shb) / len(sha | shb)
    return {"docs": [doc_a, doc_b]}, {"jaccard": round(j, 4), "hash_probes": tries,
                                      "kept": dedup.dedup([doc_a, doc_b])}


def v2_witness():
    # A 20-token span carrying a capitalised token at least every six positions, so that *every* 13-token
    # window inside it contains one.  The index stores the case-folded span, the query is raw, so no window
    # of the evaluation document can ever hit the index - although the raw overlap is verbatim.
    shared = ("the checkpoint shards were Rewritten by the compaction job before Evaluation harness ever "
              "touched them at Midnight on the cluster")
    train = ["preface material " + shared + " trailing material about unrelated things",
             "another training document with no overlap whatsoever in its body text"]
    ev = ["leading context " + shared + " and the answer continues"]
    return {"train": train, "eval": ev}, {"trick": "index is case/punctuation folded, query is raw",
                                          "shared_tokens": len(shared.split())}


def v3_witness():
    return {"n": 10, "world_size": 4, "drop_last": False}, {"trick": "pad-to-multiple repeats index n-1"}


def v6_witness():
    return {"n": 100, "seed": 7, "epoch": 0}, {"trick": "block shuffle drops the n %% 64 tail"}


BUILDERS = {"V1": v1_witness, "V2": v2_witness, "V3": v3_witness, "V6": v6_witness}
HOLDS = {"V4": "greedy packing starts a new batch before the running total can exceed max_tokens and any "
                "single document is counted as min(length, max_tokens), so no batch can overflow.",
         "V5": "the reservoir emits one item per arrival after it is full and then drains the buffer, so "
               "every input item is emitted exactly once regardless of buffer_size."}


def main(out_dir="/app"):
    wdir = os.path.join(out_dir, "witnesses")
    os.makedirs(wdir, exist_ok=True)
    findings, notes = {}, {}
    for cid, fn in BUILDERS.items():
        wit, note = fn()
        rel = "witnesses/%s.json" % cid.lower()
        json.dump(wit, open(os.path.join(out_dir, rel), "w"))
        findings[cid] = {"status": "violated", "witness": rel}
        notes[cid] = note
    for cid, arg in HOLDS.items():
        findings[cid] = {"status": "holds", "argument": arg}
    json.dump(findings, open(os.path.join(out_dir, "findings.json"), "w"), indent=1)
    print(json.dumps(notes, indent=1))
    return findings


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/app")
