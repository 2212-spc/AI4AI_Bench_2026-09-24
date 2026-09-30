"""Grade an answers.json against this task's key.

    usage: python3 grade.py <answers.json> [lab_ledger.jsonl]

The three v4 forms are graded by execution rather than by comparison with a stored answer: a
counterexample witness is rebuilt into a world and checked against the evidence, and a pre-registered
plan is executed in every world the notebook leaves open.  That needs the lab code, the hidden world and
the disclosed rows, so this grader imports the frozen copy of the package in `_lib/`.

Pass the agent's lab ledger (`<run>/lab_ledger.jsonl`) as the second argument, or set $LAB_LEDGER, so
that a counterexample witness must also reproduce the runs the agent made itself.
"""
import importlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_lib"))
from scalelab import queries as Q
from scalelab import ctx as C

items = json.load(open(os.path.join(HERE, "key.json")))
world = json.load(open(os.path.join(HERE, os.pardir, "hidden", "world.json")))
rows = json.load(open(os.path.join(HERE, "rows.json")))
ledger = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("LAB_LEDGER")

claim_fn = None
if any(it["kind"] == "cex" for it in items):
    bp = importlib.import_module("scalelab.bp." + world["module"])
    claim_fn = getattr(bp, "claim_fn", None)

gctx = C.make_ctx(world["params"], world["spec"], world["salt"], rows, items, claim_fn, ledger)
try:
    ans = json.load(open(sys.argv[1]))
except Exception as e:
    ans = {}
    print("unreadable answers:", e)
print(json.dumps(Q.grade(items, ans, gctx), indent=1))
