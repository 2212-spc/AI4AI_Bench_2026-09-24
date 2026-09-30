"""JUDGE-AUDIT (level L1): sign off a model launch from an LLM-judge eval, with a budget of human labels.  QA.

The situation.  Model B is a candidate to replace production model A.  Both answered the same 600-prompt eval
pool; an LLM judge has graded all 1,200 responses (pass / fail).  The launch review wants three answers: B's
true pass rate on the pool, the true difference B - A, and SHIP / HOLD.  "True" is defined by full expert
verification.  Three labelling services are available at very different prices: crowd votes, expert review
(a quick expert read) and full verification (definitive).  The budget buys full verification of ~16% of the
responses at most.

Difficulty mechanism (the family's signature: an instrument that everyone treats as gold is itself biased in a
way that a small random validation sample does not make visible):

  P1  THE JUDGE IS LENIENT ON LONG ANSWERS.  Wrong answers get passed more often the longer they are (the well
      known length bias of LLM judges).  B writes much longer answers than A, and B's WRONG answers are longer
      still, so the judge inflates B far more than A.  Everyone expects this and checks the judge.
  P2  THE CHECK IS BIASED THE SAME WAY.  Expert review - the obvious, cheap "gold" to validate the judge with -
      is fooled by the same long, plausible-but-wrong answers (a 2-minute read cannot catch an error buried in
      1,200 tokens).  Crowd votes share the judge's blind spot too.  Judge-vs-review agreement is high, and a
      judge corrected with review labels is still biased by 0.06-0.15 on B.
  P3  THE BLIND SPOT IS CONCENTRATED.  Review is essentially exact on A and on B's short answers.  On a random
      verification sample the review/verify disagreement rate is only a few percent - easy to read as "review is
      fine" - but all of it sits in B's long answers, and it is one-sided (always "passed but wrong").
      Precision then decides the rest: verifying at random spends most of the budget where review is already
      right (unbiased, but the +/-0.02 tolerance is met only ~30-70% of the time); locating the blind spot
      (verification vs review by length) and verifying there buys an almost exact answer.

Ground truth is exact: the pool is a finite, fixed population (correctness of each of the 1,200 responses is a
hidden bit), so the pass rates are exact means over it - no model, no sampling in the truth.  The lab returns
deterministic review / verification verdicts (one recorded verdict per response) and fresh crowd votes.
Certification (l15/tasks/judge_audit_cert via instance_gate): the reference policy passes >= 97% of Monte-Carlo
replays, every decoy policy (judge only, review as gold, crowd majority, judge corrected by review) <= 2%, and
|B - A| >= D_MIN so that the SHIP / HOLD answer is determined.
"""
import csv, io, json, math, os, re
import numpy as np
from ..core import World as _W, LabError, num, choice, rng_for

LEVEL = "L1"
N_POOL = 600
BUDGET = 500.0                 # credits
COST = {"crowd": 0.05, "review": 0.10, "verify": 2.00}
TOL = 0.02                     # Q1 and Q2 absolute tolerance
D_MIN = 0.03                   # |pass(B) - pass(A)| required for the instance (SHIP/HOLD determined)
MC_REPS = 300
REF_MIN = 0.97
DECOY_MAX = 0.02
CATS = ("code", "math", "science", "writing", "advice")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------------ the pool (deterministic from params)
_ARR = {}


def arrays(p):
    k = (p["wseed"],)
    if k in _ARR:
        return _ARR[k]
    r = np.random.default_rng([p["wseed"], 1])
    N = N_POOL
    cat = r.integers(0, len(CATS), N)
    aA = r.normal(0.7, 0.4, len(CATS))
    aB = aA + r.normal(0.0, 0.4, len(CATS))
    L0 = float(p["L0"])
    z = r.standard_normal(N)
    YA = (aA[cat] - 1.0 * z + r.logistic(0, 0.6, N) > 0).astype(int)
    YB = (aB[cat] - 1.1 * z + r.logistic(0, 0.6, N) > 0).astype(int)
    LA = np.round(200 * np.exp(0.35 * r.standard_normal(N) + 0.1 * z)).astype(int)
    LB = np.round(L0 * 0.55 * np.exp(0.45 * r.standard_normal(N) + 0.3 * z + 0.35 * (1 - YB))).astype(int)
    uf = r.random((2, N))

    def fool_j(L):   # P(judge passes a wrong answer | length)
        return 0.05 + 0.55 * _sig((L - (L0 - 250)) / 150)

    def fool_r(L):   # P(expert review passes a wrong answer | length)
        return 0.01 + 0.85 * _sig((L - L0) / 60)

    JA = np.where(YA == 1, r.random(N) < 0.95, uf[0] < fool_j(LA)).astype(int)
    JB = np.where(YB == 1, r.random(N) < 0.95, uf[1] < fool_j(LB)).astype(int)
    RA = np.where(YA == 1, r.random(N) < 0.99, uf[0] < fool_r(LA)).astype(int)
    RB = np.where(YB == 1, r.random(N) < 0.99, uf[1] < np.minimum(fool_r(LB), fool_j(LB))).astype(int)
    plaus = np.vstack([uf[0] < fool_j(LA), uf[1] < fool_j(LB)])   # "plausible-looking wrong answer"
    out = {"cat": cat, "L": {"A": LA, "B": LB}, "Y": {"A": YA, "B": YB}, "J": {"A": JA, "B": JB},
           "R": {"A": RA, "B": RB}, "plaus": {"A": plaus[0], "B": plaus[1]}}
    _ARR.clear()
    _ARR[k] = out
    return out


def truth(p):
    a = arrays(p)
    pA, pB = float(a["Y"]["A"].mean()), float(a["Y"]["B"].mean())
    return {"pass_A": pA, "pass_B": pB, "D": pB - pA, "decision": "SHIP" if pB > pA else "HOLD"}


def crowd_votes(a, m, idx, rng):
    Y, pl = a["Y"][m][idx], a["plaus"][m][idx]
    pr = np.where(Y == 1, 0.85, np.where(pl, 0.70, 0.12))
    return (rng.random(len(idx)) < pr).astype(int)


def base_params(seed, k=0):
    r = np.random.default_rng([seed, 11, k])
    return {"seed": int(seed), "k": int(k), "wseed": int(r.integers(10 ** 9)),
            "L0": float(r.choice([850.0, 1000.0, 1150.0]))}


def sample_params(seed):
    """First sub-draw whose exact B - A difference is at least D_MIN (about one draw in three)."""
    for k in range(200):
        p = base_params(seed, k)
        if abs(truth(p)["D"]) >= D_MIN:
            return p
    raise RuntimeError("no instance with |D| >= D_MIN")


# ------------------------------------------------------------------ policies (certificate)
def _grade_answer(t, q1, q2, dec):
    return (abs(q1 - t["pass_B"]) <= TOL + 1e-12) and (abs(q2 - t["D"]) <= TOL + 1e-12) and dec == t["decision"]


def policies(p, rng):
    """Each policy -> (Q1, Q2) using only what the lab would return, within the budget."""
    a = arrays(p)
    N = N_POOL
    out = {}
    JA, JB, RA, RB, YA, YB = a["J"]["A"], a["J"]["B"], a["R"]["A"], a["R"]["B"], a["Y"]["A"], a["Y"]["B"]
    # decoys: an instrument taken as gold
    out["judge_only"] = (JB.mean(), JB.mean() - JA.mean())
    out["review_all"] = (RB.mean(), RB.mean() - RA.mean())                         # 120 credits
    cA = np.mean([crowd_votes(a, "A", np.arange(N), rng) for _ in range(5)], 0) > 0.5   # 300 credits
    cB = np.mean([crowd_votes(a, "B", np.arange(N), rng) for _ in range(5)], 0) > 0.5
    out["crowd5_majority"] = (cB.mean(), cB.mean() - cA.mean())
    # judge corrected by review (PPI with review as the gold): review a random 1,000 responses
    iA = rng.choice(N, 500, replace=False); iB = rng.choice(N, 500, replace=False)
    qB = JB.mean() + (RB[iB] - JB[iB]).mean(); qA = JA.mean() + (RA[iA] - JA[iA]).mean()
    out["judge_corrected_by_review"] = (qB, qB - qA)
    # correct-but-unstratified (information only): review all + verify at random, PPI correction
    nv = int((BUDGET - 120) / COST["verify"])
    iA = rng.choice(N, nv // 2, replace=False); iB = rng.choice(N, nv - nv // 2, replace=False)
    qB = RB.mean() + (YB[iB] - RB[iB]).mean(); qA = RA.mean() + (YA[iA] - RA[iA]).mean()
    out["info_ppi_random"] = (qB, qB - qA)
    iB = rng.choice(N, nv, replace=False)
    qB = RB.mean() + (YB[iB] - RB[iB]).mean()
    out["info_ppi_random_B_only"] = (qB, qB - RA.mean())
    # reference: review everything; a 40-response random verification pilot on B; the remaining budget verifies
    # B's longest responses (where the pilot shows review disagreeing with verification); verified responses
    # count as verified, the rest as reviewed
    pil = rng.choice(N, 40, replace=False)
    rem = int((BUDGET - 120 - 40 * COST["verify"]) / COST["verify"])
    pset = set(pil.tolist())
    top = [i for i in np.argsort(-a["L"]["B"], kind="stable") if i not in pset][:rem]
    V = np.zeros(N, bool); V[pil] = True; V[top] = True
    qB = (YB[V].sum() + RB[~V].sum()) / N
    out["reference"] = (qB, qB - RA.mean())
    return out


def certify(p, reps=MC_REPS):
    t = truth(p)
    acc = {}
    for rep in range(reps):
        rng = np.random.default_rng([p["wseed"], 777, rep])
        for k, (q1, q2) in policies(p, rng).items():
            dec = "SHIP" if q2 > 0 else "HOLD"
            acc.setdefault(k, []).append((_grade_answer(t, q1, q2, dec), q1 - t["pass_B"], q2 - t["D"]))
    summ = {k: {"pass_rate": round(float(np.mean([x[0] for x in v])), 4),
                "bias_q1": round(float(np.mean([x[1] for x in v])), 4),
                "bias_q2": round(float(np.mean([x[2] for x in v])), 4),
                "sd_q2": round(float(np.std([x[2] for x in v])), 4)} for k, v in acc.items()}
    decoys = [k for k in summ if k not in ("reference",) and not k.startswith("info_")]
    ok = (abs(t["D"]) >= D_MIN and summ["reference"]["pass_rate"] >= REF_MIN
          and all(summ[k]["pass_rate"] <= DECOY_MAX for k in decoys))
    return ok, {"truth": t, "policies": summ, "decoys": decoys}


def instance_gate(p):
    ok, info = certify(p)
    return ok, info


# ------------------------------------------------------------------ lab operations
def _items(x):
    if x is None:
        raise LabError("items is required: a list of item ids, 'all', or ranges like '0-99,150,200-210'")
    if isinstance(x, str):
        s = x.strip()
        if s == "all":
            return list(range(N_POOL))
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                lo, hi = int(num(lo, "range start", 0, N_POOL - 1, True)), int(num(hi, "range end", 0, N_POOL - 1, True))
                if hi < lo:
                    raise LabError("bad range %r" % part)
                out += list(range(lo, hi + 1))
            else:
                out.append(int(num(part, "item", 0, N_POOL - 1, True)))
        return out
    if isinstance(x, (int, float)):
        return [int(num(x, "item", 0, N_POOL - 1, True))]
    if isinstance(x, list):
        return [int(num(v, "item", 0, N_POOL - 1, True)) for v in x]
    raise LabError("items must be a list of ids, 'all', or a range string")


def _cost(op):
    def f(w, a):
        its = _items(a.get("items"))
        if not its:
            raise LabError("items is empty")
        return COST[op] * len(its)
    return f


def _run(op):
    def f(w, a, ctx):
        m = choice(str(a.get("model", "")).upper(), "model", ("A", "B"))
        its = _items(a.get("items"))
        arr = arrays(w.p)
        idx = np.array(its, dtype=int)
        if op == "crowd":
            lab = crowd_votes(arr, m, idx, ctx["rng"])
        elif op == "review":
            lab = arr["R"][m][idx]
        else:
            lab = arr["Y"][m][idx]
        col = {"crowd": "crowd_vote", "review": "review_verdict", "verify": "verified_correct"}[op]
        fn = "data/%s_%s_%03d.csv" % (op, m, ctx["i"])
        buf = io.StringIO(); wr = csv.writer(buf, lineterminator="\n")
        wr.writerow(["item", "model", col])
        for i, v in zip(its, lab):
            wr.writerow([i, m, int(v)])
        if ctx.get("app_dir"):
            fp = os.path.join(ctx["app_dir"], fn)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            open(fp, "w").write(buf.getvalue())
        res = {"op": op, "model": m, "n": len(its), "file": "/app/" + fn, "n_pass": int(np.sum(lab))}
        if len(its) <= 60:
            res["labels"] = {str(i): int(v) for i, v in zip(its, lab)}
        return res
    return f


# ------------------------------------------------------------------ answers
_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"


def _line(txt, key, pat):
    m = re.findall(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)?[*_`]*%s[*_`]*\s*[:=]\s*[*_`]*\s*[*_`]*%s" % (key, pat), txt, re.M | re.I)
    if len(m) != 1:
        raise ValueError("answers.md must contain exactly one line '%s: ...' (found %d)" % (key, len(m)))
    return m[0]


def parse_answers(txt):
    q1 = float(_line(txt, "Q1", _NUM))
    q2 = float(_line(txt, "Q2", _NUM))
    q3 = _line(txt, "Q3", r"(SHIP|HOLD)\b").upper()
    if not (math.isfinite(q1) and math.isfinite(q2)):
        raise ValueError("Q1/Q2 must be finite numbers")
    if q1 > 1.0 and q1 <= 100.0:
        raise ValueError("Q1 must be a fraction in [0, 1], not a percentage")
    return q1, q2, q3


class World(_W):
    NAME = "judge_audit"
    ARTIFACTS = ["answers.md", "report.md"]
    SELF_CONSISTENT = ()
    BUDGET_UNIT = "credits"
    OPS = {"crowd": (_cost("crowd"), _run("crowd"), "one fresh crowd vote per listed response"),
           "review": (_cost("review"), _run("review"), "expert review verdict per listed response"),
           "verify": (_cost("verify"), _run("verify"), "full verification verdict per listed response")}

    def public_spec(self):
        return {"ops": {op: {"args": {"model": "A or B",
                                      "items": "list of item ids, 'all', or ranges like '0-99,150,200-210' "
                                               "(an id may be listed more than once)"},
                             "cost_per_label": COST[op], "doc": self.OPS[op][2],
                             "returns": "CSV under /app/data/ (item, model, verdict 1=correct/0=incorrect); "
                                        "labels inline when <= 60 items"} for op in ("crowd", "review", "verify")},
                "pool": "/app/data/pool.csv (%d prompts)" % N_POOL, "deliverables": ["/app/answers.md", "/app/report.md"]}

    def grade(self, art_dir, ledger=None):
        t = truth(self.p)
        items = {}
        try:
            q1, q2, q3 = parse_answers(open(os.path.join(art_dir, "answers.md")).read())
            why = None
        except Exception as ex:
            why = "answers.md: %s" % (ex if isinstance(ex, ValueError) else type(ex).__name__)
        items["R0_format"] = (why is None, why or "answers.md is well-formed")
        diag = {}
        if why is None:
            items["R1_pass_B"] = (abs(q1 - t["pass_B"]) <= TOL + 1e-12,
                                  "Q1 you %.4f, truth %.4f (tolerance %.2f)" % (q1, t["pass_B"], TOL))
            items["R2_diff"] = (abs(q2 - t["D"]) <= TOL + 1e-12,
                                "Q2 you %+.4f, truth %+.4f (tolerance %.2f)" % (q2, t["D"], TOL))
            items["R3_decision"] = (q3 == t["decision"], "Q3 you %s, truth %s" % (q3, t["decision"]))
            a = arrays(self.p)
            est = {"judge_only": (float(a["J"]["B"].mean()), float(a["J"]["B"].mean() - a["J"]["A"].mean())),
                   "review_all": (float(a["R"]["B"].mean()), float(a["R"]["B"].mean() - a["R"]["A"].mean()))}
            diag["answer"] = {"Q1": q1, "Q2": q2, "Q3": q3}
            diag["instrument_estimates"] = {k: [round(v[0], 4), round(v[1], 4)] for k, v in est.items()}
            diag["closest"] = min(list(est) + ["truth"], key=lambda k: abs((est[k][1] if k in est else t["D"]) - q2))
        else:
            for k in ("R1_pass_B", "R2_diff", "R3_decision"):
                items[k] = (False, "not evaluated")
        if ledger:
            spent = {}
            vlen = []
            a = arrays(self.p)
            for r in ledger:
                spent[r["op"]] = spent.get(r["op"], 0.0) + r["cost"]
                if r["op"] == "verify":
                    try:
                        m = str(r["args"].get("model", "")).upper()
                        its = _items(r["args"].get("items"))
                        vlen += [(m, int(a["L"][m][i])) for i in its]
                    except Exception:
                        pass
            diag["spent_by_op"] = {k: round(v, 2) for k, v in spent.items()}
            vb = [L for m, L in vlen if m == "B"]
            diag["verify_n"] = {"A": sum(1 for m, _ in vlen if m == "A"), "B": len(vb)}
            if vb:
                diag["verify_B_median_len"] = float(np.median(vb))
                diag["pool_B_median_len"] = float(np.median(a["L"]["B"]))
        n_ok = sum(1 for k, v in items.items() if v[0])
        return {"pass": bool(all(v[0] for v in items.values())), "score": round(n_ok / 4.0, 4),
                "items": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in items.items()},
                "diag": diag, "ref": t}


# ------------------------------------------------------------------ what the agent sees
def instruction(p, t):
    return """# Sign off the launch of model B

Model **B** is a candidate to replace production model **A**.  Both models answered the same eval pool of
**%d prompts** (code, math, science, writing, advice).  An LLM judge graded all %d responses (pass/fail); its
verdicts, the prompt category and each response's length in tokens are in `/app/data/pool.csv`.

Before launch the review board needs three numbers, defined by **full verification** - the only verdict the
board accepts as ground truth (see `/app/docs/labelling.md`):

* **Q1** - B's pass rate on the pool: the fraction of B's %d responses that are correct.
* **Q2** - pass rate of B minus pass rate of A on the pool.
* **Q3** - `SHIP` if B's pass rate is higher than A's, otherwise `HOLD`.

You can buy labels from the lab (`lab spec` shows the services and prices; `lab crowd|review|verify
model=A|B items=...`).  **Budget: %g credits.**  Crowd votes cost %.2f, expert review %.2f and full verification
%.2f credits per response.

## Deliverables

1. **`/app/answers.md`** with exactly these three lines (graded):

       Q1: <fraction, e.g. 0.712>
       Q2: <signed difference, e.g. -0.034>
       Q3: <SHIP or HOLD>

   Q1 and Q2 must be within %.2f of the truth; Q3 must be right.
2. **`/app/report.md`** - a note to the review board: how you estimated the numbers and how much to trust them.
   Not scored.
""" % (N_POOL, 2 * N_POOL, N_POOL, BUDGET, COST["crowd"], COST["review"], COST["verify"], TOL)


def docs(p):
    return {"docs/labelling.md": """# Labelling services

All verdicts are binary: 1 = the response is correct, 0 = it is not.

* **LLM judge** (already run on every response; `judge_A`, `judge_B` in `pool.csv`).  A strong general model
  prompted with the question and a grading rubric.
* **crowd** - %.2f credits per vote.  A vetted crowd worker reads the question and the response and votes.  Every
  vote comes from a new worker, so asking again for the same response gives an independent vote.
* **review** - %.2f credits per response.  A domain expert reads the response and records a verdict (the review
  queue budgets about two minutes per response).  The team keeps one recorded verdict per response: reviewing
  the same response again returns the same verdict (and is charged again).
* **verify** - %.2f credits per response.  Full verification: an expert independently re-derives the answer,
  runs the code, and checks every claim against sources.  This is the review board's definition of "correct".
  Verification is deterministic: verifying the same response again returns the same verdict (and is charged
  again).

Response lengths in `pool.csv` are in tokens.
""" % (COST["crowd"], COST["review"], COST["verify"])}


def starter(p):
    a = arrays(p)
    buf = io.StringIO(); wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(["item", "category", "len_A", "len_B", "judge_A", "judge_B"])
    for i in range(N_POOL):
        wr.writerow([i, CATS[a["cat"][i]], int(a["L"]["A"][i]), int(a["L"]["B"][i]), int(a["J"]["A"][i]),
                     int(a["J"]["B"][i])])
    return {"data/pool.csv": buf.getvalue()}


def hints(p):
    return {1: "## Hint\n\nBefore using any cheaper instrument as a stand-in for verification, check where (for "
               "which responses) it disagrees with verification, not just how often.\n",
            2: "## Hint\n\nExpert review, like the judge, can pass long, plausible-looking wrong answers.\n"}
