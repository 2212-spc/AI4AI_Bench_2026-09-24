"""Static consistency check between the card library and the lab backends.

A card whose parameter name does not exist in the lab it claims to describe is a latent bug: the
blueprint merges `common.draw_card` output straight into the world parameters, so a typo silently
creates an ignored key and leaves the mechanism at its default.  Cards may also legitimately name a
*summary* knob that a blueprint expands into several real keys (a spread that becomes a per-slice
vector, a per-model field, an entry inside a nested table).  Those must be declared in the card's
`derived` tuple, so that the set of undeclared mismatches is exactly the set of bugs.

Run:  cd v4 && python3 tools/check_cards.py
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalelab import cards as C
from scalelab import world as W
from scalelab import labs

PREFIX = {"C": "pretrain", "E": "evallab", "R": "rllab", "S": "servelab"}
bad = []

for cid, card in sorted(C.CARDS.items(), key=lambda kv: (kv[0][0], int(kv[0][1:]))):
    lab = PREFIX[cid[0]]
    base = W.BASE if lab == "pretrain" else labs.backend(lab).BASE
    derived = card.get("derived", ())
    if isinstance(derived, dict):
        for k, where in derived.items():
            if not str(where).strip():
                bad.append("%s: derived parameter %r does not say where it lands" % (cid, k))
        derived = set(derived)
    else:
        derived = set(derived)
    for k in card["params"]:
        if k not in base and k not in derived:
            bad.append("%s: parameter %r is not a key of the %s world and is not declared derived"
                       % (cid, k, lab))
    for k in (card.get("neutral") or {}):
        if k not in base and k not in derived:
            bad.append("%s: neutral key %r is not a key of the %s world" % (cid, k, lab))
    if not card.get("grounding"):
        bad.append("%s: no grounding" % cid)
    for k, (tb, rg) in card["params"].items():
        if rg is None:
            continue
        lo, hi = rg
        if not lo <= hi:
            bad.append("%s: range for %r is inverted" % (cid, k))
        if tb is not None and not (lo <= tb <= hi):
            bad.append("%s: textbook value %r for %r is outside the world range %r" % (cid, tb, k, rg))

# every lab must be covered, and every declared difficulty knob must be documented
for pre, lab in PREFIX.items():
    if not any(c.startswith(pre) for c in C.CARDS):
        bad.append("no cards for lab %s" % lab)
for d, spec in C.DERIVATIONS.items():
    if not spec.get("grounding"):
        bad.append("derivation %s: no grounding" % d)

# Parameter names must be unique *within* a lab (two cards writing the same world key would silently
# overwrite each other).  Across labs they may repeat - `alpha` is a scaling exponent, Gao's coefficient
# and a loss exponent in three different worlds - so anything that scans cards by parameter name has to be
# lab-scoped.  `common.box_for` is; this check records the collisions it has to survive.
by_lab = {}
for cid, card in C.CARDS.items():
    for k in card["params"]:
        by_lab.setdefault((PREFIX[cid[0]], k), []).append(cid)
for (lab, k), cids in sorted(by_lab.items()):
    if len(cids) > 1:
        bad.append("%s: parameter %r is declared by more than one card: %s" % (lab, k, ", ".join(cids)))
shared = {}
for (lab, k), cids in by_lab.items():
    shared.setdefault(k, []).extend(cids)
cross = {k: sorted(v) for k, v in shared.items() if len({PREFIX[c[0]] for c in v}) > 1}

n_by_lab = {}
for cid in C.CARDS:
    n_by_lab[PREFIX[cid[0]]] = n_by_lab.get(PREFIX[cid[0]], 0) + 1
print("cards: %s   derivations: %d   obstacles: %d   knobs: %d"
      % (n_by_lab, len(C.DERIVATIONS), len(C.OBSTACLES), len(C.KNOBS)))
print("names reused across labs (box_for must stay lab-scoped): %s"
      % ", ".join("%s=%s" % (k, "/".join(v)) for k, v in sorted(cross.items())))
if bad:
    print("\n".join("  " + b for b in bad))
    print("PROBLEMS: %d" % len(bad))
else:
    print("card library is consistent with the lab backends")
sys.exit(1 if bad else 0)
