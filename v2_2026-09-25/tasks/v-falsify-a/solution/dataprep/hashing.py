"""Deterministic 64-bit string hashing and the MinHash permutation family."""
import hashlib

P = (1 << 61) - 1


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big") % P


def perms(n_perm, tag=b"ministack-minhash-v3"):
    """n_perm affine permutations (a, c) over Z_P, derived from a fixed tag."""
    out = []
    for i in range(n_perm):
        d = hashlib.blake2b(tag + b":%d" % i, digest_size=16).digest()
        a = (int.from_bytes(d[:8], "big") % (P - 1)) + 1
        c = int.from_bytes(d[8:], "big") % P
        out.append((a, c))
    return out
