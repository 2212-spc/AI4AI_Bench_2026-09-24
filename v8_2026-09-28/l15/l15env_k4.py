"""Client library for K4 policies (mounted read-only as /lib/l15env.py inside the policy sandbox).

    env.questions   list of {"id": str, "topic": str}
    env.budget      total samples you may draw for the whole batch
    env.n_variants  number of prompt templates (variant ids 0..n_variants-1)
    env.topics      list of topic names
    env.sample(qid, variant=0) -> (answer: str, score: float)   # one generation + its verifier score; costs 1
    env.left()      -> samples remaining
    env.submit(answers: dict)                                   # {question_id: answer_string}; call exactly once

Protocol: JSON lines on stdin/stdout.  print() is redirected to stderr."""
import json, sys

_out = sys.stdout
sys.stdout = sys.stderr
_in = sys.stdin


def _send(msg):
    _out.write(json.dumps(msg) + "\n"); _out.flush()


def _call(op, **a):
    _send(dict(op=op, **a))
    line = _in.readline()
    if not line:
        raise SystemExit("host closed the connection")
    r = json.loads(line)
    if isinstance(r, dict) and r.get("error"):
        raise RuntimeError(r["error"])
    return r


def _finish():
    _send({"op": "__done__"})


class Env:
    def __init__(self):
        info = _call("init")
        self.questions = info["questions"]
        self.budget = info["budget"]
        self.n_variants = info["n_variants"]
        self.topics = info["topics"]
        self._left = self.budget

    def sample(self, qid, variant=0):
        r = _call("sample", qid=qid, variant=variant)
        self._left = r["left"]
        return r["answer"], r["score"]

    def left(self):
        return self._left

    def submit(self, answers):
        _call("submit", answers={str(k): str(v) for k, v in answers.items()})
