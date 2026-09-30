"""Mechanical verification for the three new item forms.  Copied into every exported task, so a task
can be graded with no network and no oracle answer for the parts that are checked by construction.

  cex     counterexample witness  -- the agent submits a *world* that reproduces every number it has
          seen and yet makes the claim false.  Graded by (a) consistency and (b) flip.  No oracle
          answer is needed: the grader recomputes the world from the witness.
  prereg  pre-registered plan     -- the agent submits the runs it would make and a declarative decision
          rule, before seeing any of their results.  Graded by whether the rule returns the correct
          conclusion in *every* world consistent with what was disclosed, not by the numbers it produces.
  audit   defect audit            -- the agent locates an inferential defect in a shipped analysis script
          and reports the corrected number.  Graded on the defect id, the site, and the number.

Nothing here calls a model, and nothing here depends on the agent's prose.
"""
import ast, json, math, operator

CHI2_999 = {1: 10.828, 2: 13.816, 3: 16.266, 4: 18.467, 5: 20.515, 6: 22.458, 7: 24.322, 8: 26.124,
            9: 27.877, 10: 29.588, 11: 31.264, 12: 32.909, 13: 34.528, 14: 36.123, 15: 37.697,
            16: 39.252, 17: 40.790, 18: 42.312, 19: 43.820, 20: 45.315}
Z_BAR = 3.0
# The lab rounds every reported number to six decimals, so a quantity the witness world fixes
# *exactly* (sd == 0: a count, a deterministic detector reading) is compared at that precision and
# not bit-for-bit.  Six decimals leaves no exploitable slack: a structural quantity is an exact
# function of the parameters, so matching 1e-6 on a whole row of them pins them down.
STRUCT_ATOL = 1e-6


def chi2_999(k):
    """0.999 quantile of chi-square with k degrees of freedom (Wilson-Hilferty above the table)."""
    if k <= 0:
        return 0.0
    if k in CHI2_999:
        return CHI2_999[k]
    z = 3.0902323   # Phi^{-1}(0.999)
    return k * (1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))) ** 3


# ===================================================================================== cex
class WitnessError(Exception):
    pass


def build_witness(schema, witness):
    """Validate a witness against the declared schema and return the world parameters it names.

    schema: {"params": {name: {"path": ["a","b"], "lo": x, "hi": y, "type": "float"|"int"|"choice",
                              "values": [...]}, ...}}
    Every schema parameter must be present; nothing else may be.  The path is written into a copy of
    the base parameters, so only declared parameters can differ from the true world.
    """
    ps = schema["params"]
    if not isinstance(witness, dict):
        raise WitnessError("witness must be a JSON object mapping parameter names to values")
    miss = sorted(set(ps) - set(witness))
    extra = sorted(set(witness) - set(ps))
    if miss:
        raise WitnessError("witness is missing required parameters: %s" % ", ".join(miss))
    if extra:
        raise WitnessError("witness names parameters that are not free in this question: %s" % ", ".join(extra))
    out = {}
    for k, sp in ps.items():
        v = witness[k]
        t = sp.get("type", "float")
        if t == "choice":
            if v not in sp["values"]:
                raise WitnessError("%s must be one of %s" % (k, sp["values"]))
        else:
            try:
                v = float(v)
            except Exception:
                raise WitnessError("%s must be a number" % k)
            if not math.isfinite(v):
                raise WitnessError("%s must be finite" % k)
            if not sp["lo"] - 1e-12 <= v <= sp["hi"] + 1e-12:
                raise WitnessError("%s=%g is outside the allowed range [%g, %g]" % (k, v, sp["lo"], sp["hi"]))
            if t == "int":
                v = int(round(v))
        out[k] = v
    return out


def apply_witness(base, schema, vals):
    """Return a deep copy of `base` with every witness parameter written at its declared path.

    `base` must already be a *complete* parameter set (backend.full(...)), so that a witness world is
    the true world with exactly the declared parameters moved -- nothing else, and nothing missing.
    """
    p = json.loads(json.dumps(base))
    for k, sp in schema["params"].items():
        path = sp["path"]; node = p
        for step in path[:-1]:
            if step not in node:
                raise WitnessError("this question declares a parameter at a path that does not exist: %s" % path)
            node = node[step]
        if path[-1] not in node:
            raise WitnessError("this question declares a parameter at a path that does not exist: %s" % path)
        node[path[-1]] = vals[k]
    return p


def consistency(backend, p, rows, z_bar=Z_BAR, chi_mult=1.0):
    """Check a candidate world against every disclosed row *and* every row the agent itself ran.

    Each row contributes one standardised deviation per observable.  A world passes iff
      (a) every |z| <= z_bar, and
      (b) sum z^2 <= chi2_0.999(dof).
    Returns (ok, report).
    """
    zs = []; worst = None
    for i, row in enumerate(rows):
        try:
            pred = backend.predict_row(p, row)
            obs = backend.observed(row)
        except Exception as e:
            return False, {"ok": False, "n_obs": len(zs), "sum_z2": 0.0, "chi2_bar": 0.0, "max_abs_z": 0.0,
                           "z_bar": z_bar, "row": i,
                           "reason": "row %d could not be predicted in the witness world: %s" % (i, e)}
        for k, (mu, sd) in pred.items():
            if k not in obs:
                continue
            o = float(obs[k])
            if sd <= 0:
                if abs(o - mu) > STRUCT_ATOL * max(1.0, abs(mu)):
                    return False, {"ok": False, "reason": "row %d: %s is structurally %.6g in the witness world, "
                                                          "but %.6g was observed" % (i, k, mu, o), "row": i,
                                   "n_obs": len(zs), "sum_z2": round(sum(z * z for z in zs), 4),
                                   "chi2_bar": round(chi_mult * chi2_999(len(zs)), 4),
                                   "max_abs_z": round(abs(worst[0]), 4) if worst else 0.0, "z_bar": z_bar}
                continue
            z = (o - mu) / sd
            zs.append(z)
            if worst is None or abs(z) > abs(worst[0]):
                worst = (z, i, k, o, mu, sd)
    s2 = sum(z * z for z in zs)
    bar = chi_mult * chi2_999(len(zs))
    ok = (worst is None or abs(worst[0]) <= z_bar) and s2 <= bar
    rep = {"ok": bool(ok), "n_obs": len(zs), "sum_z2": round(s2, 4), "chi2_bar": round(bar, 4),
           "max_abs_z": round(abs(worst[0]), 4) if worst else 0.0, "z_bar": z_bar}
    if worst:
        rep["worst"] = {"row": worst[1], "field": worst[2], "observed": worst[3],
                        "predicted": round(worst[4], 6), "sd": round(worst[5], 6)}
    if not ok:
        rep["reason"] = ("the witness world does not reproduce the observations"
                         if (worst and abs(worst[0]) > z_bar) else
                         "the witness world reproduces each observation but not all of them jointly")
    return bool(ok), rep


def grade_cex(item, ans, backend, base, rows, claim_fn):
    """item: {"id","kind":"cex","claim","schema","entailed": bool, ["verdict_field"]}
    ans:  {"verdict": "refutable"|"entailed", "witness": {...}}   (witness required iff refutable)
    claim_fn(p) -> bool: whether the claim holds in world p.
    """
    out = {"id": item["id"], "kind": "cex", "pass": False}
    if not isinstance(ans, dict):
        out["reason"] = "answer must be an object with keys 'verdict' and (if refutable) 'witness'"
        return out
    v = str(ans.get("verdict", "")).strip().lower()
    if v not in ("refutable", "entailed"):
        out["reason"] = "verdict must be 'refutable' or 'entailed'"
        return out
    out["verdict"] = v
    if v == "entailed":
        out["pass"] = bool(item["entailed"])
        out["reason"] = ("no consistent world falsifies the claim, as certified at build time"
                         if out["pass"] else "a consistent world that falsifies the claim exists")
        return out
    try:
        vals = build_witness(item["schema"], ans.get("witness"))
        # normalise first: a witness world must be the true world with only the declared parameters moved
        p = apply_witness(backend.full(base) if hasattr(backend, "full") else base, item["schema"], vals)
    except WitnessError as e:
        out["reason"] = "invalid witness: %s" % e
        return out
    ok, rep = consistency(backend, p, rows, z_bar=item.get("z_bar", Z_BAR))
    out["consistency"] = rep
    if not ok:
        out["reason"] = rep.get("reason", "witness world is inconsistent with the observations")
        return out
    try:
        holds = bool(claim_fn(p))
    except Exception as e:
        out["reason"] = "the claim could not be evaluated in the witness world: %s" % e
        return out
    out["claim_holds_in_witness"] = holds
    out["pass"] = not holds
    out["reason"] = ("witness is consistent with every observation and the claim is false in it"
                     if out["pass"] else
                     "witness is consistent, but the claim is still true in it")
    return out


# ===================================================================================== prereg
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow}
_CMP = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt, ast.LtE: operator.le}
_FUN = {"abs": abs, "min": min, "max": max, "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
        "mean": lambda *a: sum(a) / len(a)}


class RuleError(Exception):
    pass


def eval_expr(expr, env):
    """Evaluate a restricted arithmetic expression over named run outputs.

    Names are `<run label>.<field>` written as `r3.acc`; also bare constants and the whitelisted
    functions abs/min/max/sqrt/log/exp/mean.  Nothing else parses: no attributes beyond one level, no
    calls to anything unlisted, no comprehensions, no names with side effects.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise RuleError("expression does not parse: %s" % e)

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
                raise RuleError("only numeric constants are allowed")
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            return ev(n.operand) if isinstance(n.op, ast.UAdd) else -ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            key = "%s.%s" % (n.value.id, n.attr)
            if key not in env:
                raise RuleError("unknown quantity %r (available: %s)" % (key, ", ".join(sorted(env))))
            return float(env[key])
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise RuleError("unknown quantity %r" % n.id)
            return float(env[n.id])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN:
            if n.keywords:
                raise RuleError("keyword arguments are not allowed")
            return float(_FUN[n.func.id](*[ev(a) for a in n.args]))
        raise RuleError("expression uses a construct that is not allowed here: %s" % type(n).__name__)

    return ev(tree)


def eval_rule(rule, env):
    """rule: {"expr": "...", "cuts": [[">", 0.01, "A"], ["<=", -0.01, "B"], ["else", "undetermined"]]}
    The first matching cut wins.  A trailing ["else", label] is mandatory: a rule must return a label
    for every possible value of the statistic, in every world, or it is not a decision rule."""
    if not isinstance(rule, dict) or "expr" not in rule or "cuts" not in rule:
        raise RuleError("rule must be an object with 'expr' and 'cuts'")
    cuts = rule["cuts"]
    if not isinstance(cuts, list) or not cuts:
        raise RuleError("'cuts' must be a non-empty list")
    last = cuts[-1]
    if not (isinstance(last, list) and len(last) == 2 and last[0] == "else"):
        raise RuleError("the last cut must be [\"else\", label]: the rule has to name a conclusion for "
                        "every value the statistic could take")
    val = eval_expr(str(rule["expr"]), env)
    for i, c in enumerate(cuts):
        if not isinstance(c, list) or not c:
            raise RuleError("cut %d is malformed" % i)
        if c[0] == "else":
            if i != len(cuts) - 1 or len(c) != 2:
                raise RuleError("an 'else' cut must be last and of the form [\"else\", label]")
            return str(c[1]), val
        if len(c) != 3 or c[0] not in _CMP_NAMES:
            raise RuleError("cut %d must be [op, threshold, label] with op in %s" % (i, sorted(_CMP_NAMES)))
        if _CMP_NAMES[c[0]](val, float(c[1])):
            return str(c[2]), val
    raise RuleError("no cut matched and no 'else' cut was given")


_CMP_NAMES = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


def run_plan(session_factory, plan, labels=None):
    """Execute a plan's runs in one world and return the environment the rule is evaluated over.

    plan: {"runs": [{"label": "r1", ...request...}, ...], "rule": {...}}
    Environment keys are "<label>.<field>" for every numeric field of the result.
    """
    sess = session_factory()
    env = {}; rows = []
    for i, req in enumerate(plan["runs"]):
        req = dict(req)
        lab = str(req.pop("label", "r%d" % (i + 1)))
        res = sess.run(req)
        rows.append(res)
        for k, v in res.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                env["%s.%s" % (lab, k)] = float(v)
        for k, v in (res.get("by_order") or {}).items():
            for kk, vv in v.items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    env["%s.%s_%s" % (lab, kk, k)] = float(vv)
        for j, s in enumerate(res.get("by_slice") or []):
            for kk, vv in s.items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    env["%s.%s_s%d" % (lab, kk, j)] = float(vv)
        for j, a in enumerate(res.get("reps_acc") or []):
            env["%s.acc_r%d" % (lab, j)] = float(a)
    return env, rows, sess


def check_plan_shape(plan, spec_knobs, max_runs, budget, cost_fn):
    """Structural checks that do not need a world: shape, labels, run count, declared budget.

    `label` is metadata, not a knob: it is stripped before the request is costed or executed."""
    if not isinstance(plan, dict) or "runs" not in plan or "rule" not in plan:
        raise RuleError("plan must be an object with 'runs' (a list of requests) and 'rule'")
    runs = plan["runs"]
    if not isinstance(runs, list) or not runs:
        raise RuleError("'runs' must be a non-empty list")
    if len(runs) > max_runs:
        raise RuleError("plan has %d runs, above the limit of %d" % (len(runs), max_runs))
    seen = set(); reqs = []
    for i, r in enumerate(runs):
        if not isinstance(r, dict):
            raise RuleError("run %d must be an object" % i)
        r = dict(r)
        lab = str(r.pop("label", "r%d" % (i + 1)))
        if not lab or lab[0].isdigit() or not all(c.isalnum() or c == "_" for c in lab):
            raise RuleError("run label %r must be a name usable in the rule expression "
                            "(letters, digits, underscore; not starting with a digit)" % lab)
        if lab in seen:
            raise RuleError("duplicate run label %r" % lab)
        seen.add(lab); reqs.append(r)
    try:
        tot = sum(float(cost_fn(dict(r))) for r in reqs)
    except RuleError:
        raise
    except Exception as e:
        raise RuleError("a run in the plan is not a legal request: %s" % e)
    if tot > budget * (1 + 1e-9):
        raise RuleError("plan costs %.4g, above the declared budget %.4g" % (tot, budget))
    return tot


def grade_prereg(item, ans, worlds, session_factory_for):
    """item: {"id","kind":"prereg","max_runs","budget","labels":[...]}
    ans:  the plan.
    worlds: [{"name":…, "params":…, "label": correct conclusion, "salts":[…]}]
    session_factory_for(world, salt) -> callable returning a fresh Session.

    Pass iff the rule returns the world's correct label in every world, under every salt.
    """
    out = {"id": item["id"], "kind": "prereg", "pass": False, "by_world": []}
    try:
        cost_total = check_plan_shape(ans, item.get("knobs"), item["max_runs"], item["budget"],
                                      item["_cost_fn"])
    except RuleError as e:
        out["reason"] = str(e)
        return out
    out["plan_cost"] = cost_total
    okall = True
    for w in worlds:
        for salt in w["salts"]:
            try:
                env, rows, sess = run_plan(session_factory_for(w, salt), ans)
                got, val = eval_rule(ans["rule"], env)
            except Exception as e:
                out["by_world"].append({"world": w["name"], "salt": salt, "error": str(e)})
                okall = False
                continue
            good = (got == w["label"])
            okall = okall and good
            out["by_world"].append({"world": w["name"], "salt": salt, "expected": w["label"],
                                    "got": got, "stat": round(val, 6), "ok": bool(good)})
    out["pass"] = bool(okall)
    if not okall:
        bad = [b for b in out["by_world"] if not b.get("ok")]
        out["reason"] = ("the rule does not return the correct conclusion in every world consistent with "
                         "what was disclosed (%d of %d world/salt combinations wrong)"
                         % (len(bad), len(out["by_world"])))
    else:
        out["reason"] = "the plan discriminates in every consistent world, under every salt"
    return out


# ===================================================================================== audit
def grade_audit(item, ans):
    """item: {"id","kind":"audit","defect","sites":[["file",lo,hi],…],"lo","hi"[,"accept_defects"]}
    ans:  {"defect": id, "site": "file:line", "corrected": number | {"lo":…,"hi":…}}
    All three parts must be right.  `sites` is the span of lines that count as the defect's location.
    """
    out = {"id": item["id"], "kind": "audit", "pass": False, "parts": {}}
    if not isinstance(ans, dict):
        out["reason"] = "answer must be an object with keys 'defect', 'site' and 'corrected'"
        return out
    acc = set(item.get("accept_defects") or [item["defect"]])
    got = str(ans.get("defect", "")).strip()
    out["parts"]["defect"] = got in acc
    site = str(ans.get("site", "")).strip()
    ok_site = False
    if ":" in site:
        f, _, ln = site.rpartition(":")
        try:
            ln = int(str(ln).strip())
        except Exception:
            ln = None
        for sf, lo, hi in item["sites"]:
            if ln is not None and f.strip().endswith(sf) and lo <= ln <= hi:
                ok_site = True
    out["parts"]["site"] = ok_site
    c = ans.get("corrected")
    if isinstance(c, dict):
        try:
            lo, hi = float(c["lo"]), float(c["hi"])
        except Exception:
            lo = hi = None
        num_ok = lo is not None and lo <= item["hi"] and hi >= item["lo"] and (hi - lo) <= item["max_width"]
    else:
        try:
            x = float(c)
            num_ok = item["lo"] <= x <= item["hi"]
        except Exception:
            num_ok = False
    out["parts"]["corrected"] = bool(num_ok)
    out["pass"] = all(out["parts"].values())
    if not out["pass"]:
        missing = [k for k, v in out["parts"].items() if not v]
        out["reason"] = "wrong or missing: %s" % ", ".join(missing)
    return out
