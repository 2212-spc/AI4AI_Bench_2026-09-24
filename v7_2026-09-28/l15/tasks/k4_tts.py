"""K4 tts-policy: write a test-time sampling policy for a math-QA service with an imperfect verifier.

Mechanism (hidden; transplanted from test-time-compute / verifier literature - self-consistency, best-of-N with
reward models, reward-model exploitation, prompt-ensembling, adaptive allocation - with counterfactual constants):
  * each question has a topic, a solve rate p, and - with probability trap_rate - "attractor" wrong answers that
    the generator produces consistently.  Attractors are specific to the prompt template (variant) except for a
    `shared` fraction of questions where all templates share one misconception;
  * the verifier score of a sample is Normal(mu, sig) with mu = 1.0 for correct answers, mu_w for scattered wrong
    answers, and mu_att[topic] for attractors: on some topics the verifier sees through attractors (mu_att < 0), on
    others it is fooled (mu_att > 1, i.e. it prefers the attractor to the truth);
  * a batch of Q questions shares a global sample budget of 6 per question.
The agent studies labelled dev samples, then submits /app/policy.py; it is graded on hidden test batches (fresh
questions, same service) against the majority-vote baseline and a reference policy, with common random numbers.
"""
import json, math, os
import numpy as np
from ..core import World as _W, LabError, num, rng_for
from .. import runner

TOP = ["algebra", "geometry", "combinatorics", "number_theory"]
NV = 5
Q_TEST = 400
PER_Q = 6
TEST_WORLDS = ["T0", "T1", "T2"]
TAU = 0.85
EVAL_COST = 1000
BUDGET = 10000
HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(os.path.dirname(HERE), "l15env_k4.py")


def sample_params(seed):
    g = np.random.default_rng(2000 + seed)
    fooled = sorted(int(x) for x in g.choice(4, size=2, replace=False))
    return {"mu_att": [round(float(g.uniform(1.45, 1.8)), 3) if i in fooled else round(float(g.uniform(-0.8, -0.3)), 3) for i in range(4)],
            "mu_w": -0.6, "sig": round(float(g.uniform(0.33, 0.40)), 3),
            "trap_rate": round(float(g.uniform(0.45, 0.6)), 3), "shared": round(float(g.uniform(0.45, 0.65)), 3),
            "amass": [0.55, 0.9], "p_beta": [2.0, 2.2], "trap_p_factor": 0.5, "fooled": [TOP[i] for i in fooled]}


# ----------------------------------------------------------------------------------------------- simulator
class Batch:
    """One batch of questions (a 'world' in the grading sense).  Fully determined by (salt, wid)."""

    def __init__(self, p, salt, wid, Q):
        self.p, self.salt, self.wid, self.Q = p, salt, wid, Q
        self.qs = []
        for i in range(Q):
            g = rng_for(salt, "q", wid, i)
            t = int(g.integers(4))
            pc = float(np.clip(g.beta(*p["p_beta"]), 0.03, 0.95))
            trap = bool(g.random() < p["trap_rate"])
            a = [float(g.uniform(*p["amass"])) if trap else float(g.uniform(0, 0.1)) for _ in range(NV)]
            if trap:
                pc *= p["trap_p_factor"]
            shared = bool(g.random() < p["shared"])
            ans = g.choice(np.arange(10, 10000), size=1 + NV, replace=False)
            self.qs.append({"id": "%s-%03d" % (wid, i), "topic": TOP[t], "p": pc, "a": a, "shared": shared, "trap": trap,
                            "correct": str(int(ans[0])), "att": [str(int(x)) for x in ans[1:]]})
        self.by_id = {q["id"]: q for q in self.qs}
        self.draws = {}

    def draw(self, qid, v):
        q = self.by_id[qid]
        j = self.draws.get((qid, v), 0); self.draws[(qid, v)] = j + 1
        g = rng_for(self.salt, "s", self.wid, qid, v, j)
        u = g.random(); pc = q["p"]; pa = q["a"][v] * (1 - pc)
        if u < pc:
            ans, mu = q["correct"], 1.0
        elif u < pc + pa:
            ans, mu = (q["att"][0] if q["shared"] else q["att"][v]), self.p["mu_att"][TOP.index(q["topic"])]
        else:
            ans, mu = str(int(g.integers(10, 100000))), self.p["mu_w"]
            if ans == q["correct"]:
                mu = 1.0
        return ans, round(float(g.normal(mu, self.p["sig"])), 4)


class Driver:
    """Host side of the policy protocol (also used in-process via LocalEnv)."""

    def __init__(self, batch, budget):
        self.b = batch; self.left = int(budget); self.finished = False; self.answers = {}; self.used = 0

    def handle(self, m):
        op = m.get("op")
        if op == "init":
            return {"questions": [{"id": q["id"], "topic": q["topic"]} for q in self.b.qs], "budget": self.left,
                    "n_variants": NV, "topics": TOP}
        if op == "sample":
            qid = m.get("qid"); v = m.get("variant", 0)
            if qid not in self.b.by_id:
                return {"error": "unknown question id %r" % (qid,)}
            if not isinstance(v, int) or not 0 <= v < NV:
                return {"error": "variant must be an int in 0..%d" % (NV - 1)}
            if self.left <= 0:
                return {"error": "sample budget exhausted"}
            self.left -= 1; self.used += 1
            a, s = self.b.draw(qid, v)
            return {"answer": a, "score": s, "left": self.left}
        if op == "left":
            return {"left": self.left}
        if op == "submit":
            ans = m.get("answers")
            if not isinstance(ans, dict):
                return {"error": "answers must be a dict {question_id: answer_string}"}
            self.answers = {str(k): str(v) for k, v in ans.items()}
            self.finished = True
            return {"ok": True}
        return {"error": "unknown op %r" % (op,)}

    def accuracy(self, per_topic=False):
        ok = [self.answers.get(q["id"]) == q["correct"] for q in self.b.qs]
        acc = float(np.mean(ok))
        if not per_topic:
            return acc
        return acc, {t: round(float(np.mean([o for o, q in zip(ok, self.b.qs) if q["topic"] == t] or [0])), 4) for t in TOP}


class LocalEnv:
    """In-process twin of the sandbox client (for reference policies and baselines)."""

    def __init__(self, drv):
        self._d = drv
        info = drv.handle({"op": "init"})
        self.questions, self.budget, self.n_variants, self.topics = info["questions"], info["budget"], info["n_variants"], info["topics"]

    def sample(self, qid, variant=0):
        r = self._d.handle({"op": "sample", "qid": qid, "variant": variant})
        if "error" in r:
            raise RuntimeError(r["error"])
        return r["answer"], r["score"]

    def left(self):
        return self._d.left

    def submit(self, answers):
        r = self._d.handle({"op": "submit", "answers": answers})
        if "error" in r:
            raise RuntimeError(r["error"])


# ----------------------------------------------------------------------------------------------- reference policies (source text)
POL_COMMON = '''
import math
def _agg(samples, weight):
    c = {}
    for a, s in samples:
        c[a] = c.get(a, 0.0) + weight(s)
    return max(c, key=c.get) if c else "0"
'''
POL_FIXED = POL_COMMON + '''
VARIANTS = %s          # template schedule
WEIGHT = %r            # "one" | "exp" | "hard"
def run(env):
    out = {}
    per = env.budget // len(env.questions)
    for q in env.questions:
        ss = [env.sample(q["id"], VARIANTS[k %% len(VARIANTS)]) for k in range(per)]
        w = {"one": lambda s: 1.0, "exp": math.exp, "hard": lambda s: math.exp(20 * s)}[WEIGHT]
        out[q["id"]] = _agg(ss, w)
    env.submit(out)
'''
POL_TOPIC = POL_COMMON + '''
TRUST = %r
def run(env):
    out = {}
    per = env.budget // len(env.questions)
    for q in env.questions:
        ss = [env.sample(q["id"], k %% 5) for k in range(per)]
        w = (lambda s: math.exp(20 * s)) if TRUST.get(q["topic"]) else (lambda s: 1.0)
        out[q["id"]] = _agg(ss, w)
    env.submit(out)
'''
POL_REF = POL_COMMON + '''
TRUST = %r   # topic -> does the verifier see through consistent wrong answers?
def run(env):
    out = {}
    qs = env.questions
    left = env.budget
    for i, q in enumerate(qs):
        reserve = 2 * (len(qs) - i - 1)
        cap = max(1, min(12, left - reserve))
        ss = []
        if TRUST.get(q["topic"]):
            while len(ss) < min(cap, 10):
                a, s = env.sample(q["id"], len(ss) %% 5); ss.append((a, s))
                if s > 0.8:
                    break
            out[q["id"]] = _agg(ss, lambda s: math.exp(20 * s))
        else:
            while len(ss) < cap:
                a, s = env.sample(q["id"], len(ss) %% 5); ss.append((a, s))
                c = {}
                for a2, _ in ss:
                    c[a2] = c.get(a2, 0) + 1
                top = sorted(c.values(), reverse=True) + [0]
                if len(ss) >= 3 and top[0] >= 3 and top[0] - top[1] >= 2:
                    break
            out[q["id"]] = _agg(ss, lambda s: 1.0)
        left -= len(ss)
    env.submit(out)
'''


POL_XVAR = POL_COMMON + '''
RELIABLE = %r
def run(env):
    qs = {q["id"]: q["topic"] for q in env.questions}; order = list(qs)
    S = {k: [] for k in qs}
    def dec(qid):
        by = {}
        for v, a, s in S[qid]:
            e = by.setdefault(a, [set(), 0, 0.0]); e[0].add(v); e[1] += 1; e[2] += s
        if not by:
            return None, False
        multi = {a: e for a, e in by.items() if len(e[0]) >= 2}
        if multi:
            return max(multi, key=lambda a: (len(multi[a][0]), multi[a][2])), True
        if qs[qid] in RELIABLE:
            b = max(by, key=lambda a: by[a][2] / by[a][1])
            return b, by[b][2] / by[b][1] > 0.3
        return max(by, key=lambda a: by[a][1]), False
    for qid in order:
        for k in range(2):
            a, s = env.sample(qid, k); S[qid].append((k, a, s))
    while env.left() > 0:
        op = [q for q in order if not dec(q)[1] and len(S[q]) < 30]
        if not op:
            break
        op.sort(key=lambda q: len(S[q]))
        for q in op:
            if env.left() <= 0:
                break
            v = len(S[q]) %% 5
            a, s = env.sample(q, v); S[q].append((v, a, s))
    env.submit({q: (dec(q)[0] or "") for q in order})
'''


POL_LLR = POL_COMMON + '''
W = %r      # topic -> [(upper_edge, weight), ...]: log-odds that a sample with this score is correct
DIV = %r; ADAPT = %r; THR = %r
def w(t, s):
    for hi, v in W[t]:
        if s < hi:
            return v
    return W[t][-1][1]
def run(env):
    out = {}; qs = env.questions; left = env.budget
    for i, q in enumerate(qs):
        reserve = 2 * (len(qs) - i - 1) if ADAPT else 0
        cap = max(1, min(14, left - reserve)) if ADAPT else env.budget // len(qs)
        tot = {}; n = 0
        while n < cap:
            a, s = env.sample(q["id"], (n %% 5) if DIV else 0); n += 1
            tot[a] = tot.get(a, 0.0) + w(q["topic"], s)
            if ADAPT:
                srt = sorted(tot.values(), reverse=True) + [0.0]
                if srt[0] >= THR and srt[0] - max(srt[1], 0) >= THR:
                    break
        out[q["id"]] = max(tot, key=tot.get)
        left -= n
    env.submit(out)
'''
EDGES = [round(-1.0 + 0.2 * i, 2) for i in range(18)]


def _pdf(x, m, s):
    return math.exp(-0.5 * ((x - m) / s) ** 2) / s


def true_bins(p, t, mode="true"):
    ti = TOP.index(t); sig = p["sig"]; ma = p["mu_att"][ti]; out = []
    for e in EDGES:
        x = e - 0.1
        v = math.log(.35 * _pdf(x, 1, sig) / (.3 * _pdf(x, ma, sig) + .35 * _pdf(x, p["mu_w"], sig) + 1e-300))
        out.append((e, round(float(np.clip(v, -4, 4)), 3)))
    if mode == "mono":
        m = -9.0; out = [(e, (m := max(m, v))) for e, v in out]
    if mode == "flat":
        out = [(e, 1.0) for e, v in out]
    return out


def llr_src(p, adapt=True, div=True, modes=None, W=None):
    W = W or {t: true_bins(p, t, (modes or {}).get(t, "true")) for t in TOP}
    return POL_LLR % (W, div, adapt, 2.0)


def learn_bins(samples):
    """samples: [(score, is_correct)] -> smoothed per-bin log-odds (what a careful analyst would compute)."""
    out = []; lo = -1e9
    for e in EDGES:
        inb = [c for s, c in samples if lo <= s < e]; lo = e
        c = sum(inb); w = len(inb) - c
        out.append((e, round(float(np.clip(math.log((c + .5) / (w + .5)), -4, 4)), 3)))
    return out


def pol_src(kind, p=None, trust=None):
    if kind == "maj":
        return POL_FIXED % ("[0]", "one")
    if kind == "bon":
        return POL_FIXED % ("[0]", "hard")
    if kind == "wvote":
        return POL_FIXED % ("[0]", "exp")
    if kind == "div_wvote":
        return POL_FIXED % ("[0, 1, 2, 3, 4]", "exp")
    if kind == "div_bon":
        return POL_FIXED % ("[0, 1, 2, 3, 4]", "hard")
    if kind == "topic_fixed":
        return POL_TOPIC % (trust or {t: t not in p["fooled"] for t in TOP})
    if kind == "xvar":
        return POL_XVAR % (tuple(t for t in TOP if t not in p["fooled"]),)
    if kind == "ref":
        return llr_src(p)
    if kind == "llr_noadapt":
        return llr_src(p, adapt=False)
    if kind == "llr_mono":
        return llr_src(p, modes={t: "mono" for t in p["fooled"]})
    if kind == "llr_flatF":
        return llr_src(p, modes={t: "flat" for t in p["fooled"]})
    if kind == "ref_heur":
        return POL_REF % (trust or {t: t not in p["fooled"] for t in TOP})
    if kind == "ref_alltrust":
        return POL_REF % ({t: True for t in TOP})
    if kind == "ref_notrust":
        return POL_REF % ({t: False for t in TOP})
    raise KeyError(kind)


def run_local(src, p, salt, wid, Q=Q_TEST):
    ns = {}
    exec(compile(src, "<policy>", "exec"), ns)
    drv = Driver(Batch(p, salt, wid, Q), PER_Q * Q)
    ns["run"](LocalEnv(drv))
    return drv.accuracy()


def truth(p, salt):
    out = {}
    for k in ("maj", "ref"):
        out[k] = [run_local(pol_src(k, p), p, salt, w) for w in TEST_WORLDS]
    return out


def run_sandboxed(sub_dir, p, salt, wid, Q=Q_TEST, timeout=120):
    drv = Driver(Batch(p, salt, wid, Q), PER_Q * Q)
    if not os.path.exists(os.path.join(sub_dir, "policy.py")):
        return None, {"ok": False, "error": "missing policy.py"}
    r = runner.run(sub_dir, "policy", "run", LIB_PATH, drv, timeout)
    if not r["ok"]:
        return None, r
    acc, per = drv.accuracy(per_topic=True)
    r["per_topic"] = per; r["samples_used"] = drv.used
    return acc, r


# ----------------------------------------------------------------------------------------------- lab ops
def _cost_dev(w, a):
    n = num(a.get("n", 10), "n", 1, 50, integer=True)
    vs = a.get("variants", [0])
    if not isinstance(vs, list) or not vs or any((not isinstance(v, int)) or v < 0 or v >= NV for v in vs):
        raise LabError("variants must be a non-empty list of ints in 0..%d" % (NV - 1))
    k = num(a.get("k", 1), "k", 1, 8, integer=True)
    return float(n * len(vs) * k)


def _run_dev(w, a, ctx):
    n = int(a.get("n", 10)); vs = a.get("variants", [0]); k = int(a.get("k", 1))
    topic = a.get("topic")
    if topic is not None and topic not in TOP:
        raise LabError("topic must be one of %s" % TOP)
    # fresh labelled dev questions; if a topic is requested, draw from that topic only
    b = Batch(w.p, w.salt, "D%d" % ctx["i"], 400)
    pool = [q for q in b.qs if topic is None or q["topic"] == topic][:n]
    out = []
    for q in pool:
        ss = []
        for v in vs:
            for _ in range(k):
                ans, s = b.draw(q["id"], v)
                ss.append({"variant": v, "answer": ans, "score": s, "correct": ans == q["correct"]})
        out.append({"id": q["id"], "topic": q["topic"], "correct_answer": q["correct"], "samples": ss})
    return {"questions": out}


def _cost_eval(w, a):
    return float(EVAL_COST)


def _run_eval(w, a, ctx):
    app = ctx["app_dir"]
    if not app:
        raise LabError("no submission directory")
    acc, r = run_sandboxed(app, w.p, w.salt, "E%d" % ctx["i"])
    return {"accuracy": None if acc is None else round(acc, 4), "per_topic_accuracy": r.get("per_topic"),
            "samples_used": r.get("samples_used"), "sample_budget": PER_Q * Q_TEST, "questions": Q_TEST,
            "error": r.get("error"), "stderr_tail": (r.get("stderr_tail") or "")[-1500:]}


class World(_W):
    NAME = "k4_tts"
    ARTIFACTS = ["policy.py"]
    BUDGET_UNIT = "credits"
    OPS = {"dev_sample": (_cost_dev, _run_dev, "labelled samples on fresh dev questions"),
           "evaluate": (_cost_eval, _run_eval, "run /app/policy.py on a fresh dev batch")}

    def public_spec(self):
        return {"ops": {
            "dev_sample": {"args": {"n": "number of fresh dev questions (1..50)", "topic": "optional: one of %s" % TOP,
                                    "variants": "list of prompt templates to sample from (ints 0..4), default [0]",
                                    "k": "samples per template per question (1..8), default 1"},
                           "cost": "n * len(variants) * k credits",
                           "returns": "per question: topic, correct_answer, samples [{variant, answer, score, correct}]"},
            "evaluate": {"args": {}, "cost": "%d credits" % EVAL_COST,
                         "returns": "accuracy of /app/policy.py on a fresh batch of %d dev questions with %d samples/question "
                                    "budget; per-topic accuracy; samples used; error/stderr if it crashed" % (Q_TEST, PER_Q)}},
            "deliverable": "/app/policy.py (see /app/docs/policy_api.md)"}

    def grade(self, art_dir, ledger=None):
        t = truth(self.p, self.salt)
        accs, errs, per = [], [], []
        for wi, wid in enumerate(TEST_WORLDS):
            acc, r = run_sandboxed(art_dir, self.p, self.salt, wid)
            if acc is None:
                errs.append("%s: %s" % (wid, r.get("error"))); accs.append(0.0)
            else:
                accs.append(acc); per.append(r.get("per_topic"))
        base = float(np.mean(t["maj"])); ref = float(np.mean(t["ref"]))
        acc = float(np.mean(accs))
        s = (acc - base) / (ref - base)
        items = {"R0_runs": (not errs, "; ".join(errs) or "policy ran and submitted on all %d test batches within budget" % len(TEST_WORLDS)),
                 "R1_accuracy": (s >= TAU, "test accuracy %.4f (majority@6 %.4f, reference %.4f) -> s=%.3f, need %.2f" % (acc, base, ref, s, TAU))}
        diag = {"acc_by_world": [round(a, 4) for a in accs], "per_topic": per}
        if ledger is not None:
            diag["n_dev_samples"] = int(sum(r["cost"] for r in ledger if r["op"] == "dev_sample"))
            diag["n_evaluate"] = sum(1 for r in ledger if r["op"] == "evaluate")
        return {"pass": all(v[0] for v in items.values()), "score": round(s, 4),
                "items": {k: {"ok": v[0], "detail": v[1]} for k, v in items.items()}, "diag": diag,
                "ref": {"maj": t["maj"], "ref": t["ref"]}}


# ----------------------------------------------------------------------------------------------- scripted strategies (gates)
def _write_pol(art_dir, src):
    open(os.path.join(art_dir, "policy.py"), "w").write(src)


def strat_oracle(sess, art_dir, rng):
    """Existence proof: per-topic calibration curves (score -> P(correct)) from labelled dev samples, then a
    calibrated weighted vote with sequential early stopping."""
    W = {}
    for t in TOP:
        S = []
        for _ in range(3):
            r = sess.call("dev_sample", {"n": 50, "topic": t, "variants": [0, 1, 2, 3, 4], "k": 1})
            S += [(x["score"], x["correct"]) for q in r["questions"] for x in q["samples"]]
        W[t] = learn_bins(S)
    _write_pol(art_dir, llr_src(sess.w.p, W=W))
    sess.call("evaluate", {})


def _fixed(kind):
    def f(sess, art_dir, rng):
        _write_pol(art_dir, pol_src(kind, sess.w.p))
    return f


def strat_crash(sess, art_dir, rng):
    _write_pol(art_dir, "def run(env):\n    raise ValueError('boom')\n")


def strat_cheat_probe(sess, art_dir, rng):
    """Tries to escape: read hidden/app files, open a socket, over-spend; then answers by majority."""
    _write_pol(art_dir, '''
import os, socket
def run(env):
    leaks = []
    for p in ["/app", "/app/.lab_token", "/hidden", "/tmp", os.path.expanduser("~")]:
        try: leaks.append((p, os.listdir(p) if os.path.isdir(p) else open(p).read()[:20]))
        except Exception as e: leaks.append((p, type(e).__name__))
    try:
        s = socket.create_connection(("127.0.0.1", 18787), timeout=1); leaks.append("net-ok")
    except Exception as e: leaks.append(("net", type(e).__name__))
    print(leaks)
    out = {}
    for q in env.questions:
        try:
            a = [env.sample(q["id"], 0) for _ in range(50)]
        except RuntimeError:
            a = [("0", 0)]
        out[q["id"]] = a[0][0]
    env.submit(out)
''')


def strat_nop(sess, art_dir, rng):
    pass


STRATEGIES = {"oracle": (strat_oracle, "pass"), "maj": (_fixed("maj"), "fail"), "bon": (_fixed("bon"), "fail"),
              "div_wvote": (_fixed("div_wvote"), "fail"), "topic_fixed": (_fixed("topic_fixed"), "fail"),
              "ref_heur": (_fixed("ref_heur"), "fail"), "llr_noadapt": (_fixed("llr_noadapt"), "fail"),
              "llr_mono": (_fixed("llr_mono"), "fail"), "llr_flatF": (_fixed("llr_flatF"), "fail"), "xvar": (_fixed("xvar"), "fail"), "crash": (strat_crash, "fail"),
              "cheat_probe": (strat_cheat_probe, "fail"), "nop": (strat_nop, "fail")}


def instance_gate(p, salt="screen"):
    """Noiseless-ish screen in-process: reference must beat every simple policy by a margin."""
    res = {}
    for k in ("maj", "bon", "div_wvote", "topic_fixed", "ref_heur", "llr_noadapt", "llr_mono", "llr_flatF", "xvar", "ref"):
        res[k] = float(np.mean([run_local(pol_src(k, p), p, salt, w) for w in TEST_WORLDS]))
    b, r = res["maj"], res["ref"]
    s = {k: round((v - b) / (r - b), 3) for k, v in res.items()}
    worst = max(v for k, v in s.items() if k not in ("ref",))
    ok = (r - b) >= 0.3 and worst <= TAU - 0.07
    return ok, {"gap": round(r - b, 4), "best_simple_s": worst, **{"s_" + k: v for k, v in s.items()}}


# ----------------------------------------------------------------------------------------------- agent-facing text
def instruction(p, t):
    return """# Test-time sampling policy for a math-answering service

`solver-7b` answers competition-style math questions. It is stochastic: you can sample it several times per
question, and a separate **verifier model** scores every candidate answer (higher = the verifier believes the
answer more). Each question can be asked through one of **5 interchangeable prompt templates** (`variant`
0-4); the templates are paraphrases of the same instruction, and the service treats them as equivalent.

Serving is capacity-limited: for a batch of %d questions you get a pool of **%d samples in total** (6 per
question on average). The pool is shared - you may spend 2 samples on one question and 12 on another - and the
service currently just takes a plain majority vote over 6 samples of template 0. You have been asked to
replace that with a better policy.

## Your job

Write **`/app/policy.py`**, a Python file exposing `run(env)` (see `/app/docs/policy_api.md`; a working
majority-vote implementation is already there). It receives the batch, may call `env.sample(qid, variant)` up
to the pooled budget, and must call `env.submit({question_id: answer_string})` exactly once. It runs in a
sandbox with no network and no access to this directory - only the standard library, and `numpy` is *not*
available.

## The lab

Run `/app/bin/lab spec` first. You have %d credits.

* `lab dev_sample n=20 topic=algebra variants='[0,1,2,3,4]' k=2` - draws fresh **labelled** dev questions from
  the same service and returns, for each, the ground-truth answer and the requested samples with their
  verifier scores. Costs `n * len(variants) * k` credits.
* `lab evaluate` - runs your current `/app/policy.py` on a fresh batch of %d dev questions under the real
  budget and returns its accuracy overall and per topic (plus stderr if it crashes). Costs %d credits, so you
  can afford a handful of these - use dev samples to understand the service, not `evaluate` as a search loop.

Example:

    /app/bin/lab dev_sample n=15 topic=geometry variants='[0,1,2]' k=2

## How it is graded (mechanically)

Your `policy.py` is run on **%d fresh hidden batches** of %d questions from the same service, with the same
pooled budget, and scored by

    s = (your accuracy - majority@6 accuracy) / (reference policy accuracy - majority@6 accuracy)

where both reference numbers are measured on those same hidden batches. You pass if

1. **R0** - the policy runs to completion and submits an answer for every question on all %d batches, without
   crashing and without exceeding the sample pool, and
2. **R1** - `s >= %.2f`.

On the dev distribution the plain majority@6 baseline scores around %.2f and the reference policy around
%.2f accuracy, so R1 needs roughly **%.2f** accuracy or better. Both bars are re-measured on the hidden
batches, so aim to beat the reference, not to match a number.

Only `/app/policy.py` is graded. Write it, make sure `lab evaluate` runs it without errors, then stop.
""" % (Q_TEST, PER_Q * Q_TEST, BUDGET, Q_TEST, EVAL_COST, len(TEST_WORLDS), Q_TEST, len(TEST_WORLDS), TAU,
        t.get("acc_maj", 0.0), t.get("acc_ref", 0.0),
        t.get("acc_maj", 0.0) + TAU * (t.get("acc_ref", 0.0) - t.get("acc_maj", 0.0)))


def instance_truth(cfg):
    """Salt-dependent reference accuracies, cached at build time so the instruction can quote real numbers."""
    tt = truth(cfg["params"], cfg["salt"])
    return {"acc_maj": round(float(np.mean(tt["maj"])), 4), "acc_ref": round(float(np.mean(tt["ref"])), 4)}


def docs(p):
    return {"docs/policy_api.md": """# Policy API

Your file `/app/policy.py` must define `run(env)`. It is imported and called once per batch inside a sandbox
(no network, no filesystem access beyond your own file, standard library only - **no numpy**). `print()` goes
to stderr and is shown to you by `lab evaluate` when something goes wrong.

    env.questions     # [{"id": "T0-017", "topic": "algebra"}, ...] - the whole batch, in a fixed order
    env.budget        # total samples available for the whole batch (shared pool)
    env.n_variants    # 5
    env.topics        # ["algebra", "geometry", "combinatorics", "number_theory"]

    answer, score = env.sample(qid, variant=0)
        # one fresh generation for that question through that prompt template, plus its verifier score.
        # answer is a string; score is a float. Costs 1 from the pool. Raises RuntimeError when the pool is
        # empty, so keep track: env.left() returns what is left.

    env.submit({qid: answer_string, ...})
        # call exactly once, at the end. Questions you omit count as wrong.

Minimal example (this is the current production policy):

    def run(env):
        out = {}
        per = env.budget // len(env.questions)
        for q in env.questions:
            counts = {}
            for _ in range(per):
                a, s = env.sample(q["id"], 0)
                counts[a] = counts.get(a, 0) + 1
            out[q["id"]] = max(counts, key=counts.get)
        env.submit(out)

Notes on the service (from the serving team):

* the verifier is a separately trained reward model. It was trained on a mixture of human preference data and
  solution traces, and nobody has audited it per topic;
* the 5 prompt templates were written by different people and are considered interchangeable; the service has
  always used template 0;
* answers are compared as exact strings after normalisation, so two samples agree iff their strings match.
""",
            "docs/service_notes.md": """# solver-7b serving notes

* batch size %d questions, pooled sampling budget %d (6/question), temperature fixed at 0.8
* topics are tagged upstream and are reliable
* current policy: majority vote over 6 samples of template 0 (`policy.py` in this directory)
* known issue tickets: "verifier score seems to disagree with graders on some topics" (open),
  "paraphrasing the prompt changes which wrong answer the model settles on" (open, not reproduced)
""" % (Q_TEST, PER_Q * Q_TEST)}


def starter(p):
    return {"policy.py": """# Current production policy: majority vote over 6 samples of template 0.
# Replace this with your own. See /app/docs/policy_api.md.


def run(env):
    out = {}
    per = env.budget // len(env.questions)
    for q in env.questions:
        counts = {}
        for _ in range(per):
            a, s = env.sample(q["id"], 0)
            counts[a] = counts.get(a, 0) + 1
        out[q["id"]] = max(counts, key=counts.get)
    env.submit(out)
"""}


def hints(p):
    return {1: "\n## Hint\nThe verifier score is a measurement, not a verdict. Before you trust it, check what it is "
               "worth - and whether it is worth the same everywhere.\n",
            2: "\n## Hint\nOn some topics the model produces the *same* wrong answer again and again, and the verifier "
               "likes it better than the truth; on other topics the verifier is a reliable signal. The prompt templates "
               "mostly disagree about which wrong answer that is. Calibrate score -> P(correct) per topic on labelled "
               "dev data, and remember the sample pool is shared across questions.\n"}
