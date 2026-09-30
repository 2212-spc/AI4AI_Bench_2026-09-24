"""Verifier core for family A1 (shared by generator certificates and tests/)."""
import importlib, json, os, sys, traceback
import numpy as np


def teacher_from(d):
    return {k: np.array(v) if isinstance(v, list) else v for k, v in d.items()}


def sample(teacher, n, seed):
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, teacher["d_in"]))
    f = np.tanh(x @ teacher["W1"] + teacher["b1"]) @ teacher["W2"]
    y = f + teacher["noise"] * r.standard_normal(n)
    return x, y


def ref_mse(p, x, y):
    h = x @ p["W1"] + p["b1"]
    out = np.maximum(h, 0.0) @ p["W2"] + p["b2"]
    return float(np.mean((out.reshape(-1) - y) ** 2))


def import_train(app_dir):
    for m in list(sys.modules):
        if m == "minilab" or m.startswith("minilab."):
            del sys.modules[m]
    sys.path.insert(0, app_dir)
    try:
        mod = importlib.import_module("minilab.trainer")
    finally:
        sys.path.pop(0)
    return mod.train


def evaluate(app_dir, spec, teacher, verbose=True):
    """Returns (passed, report). spec['checks'] is a list of configs with reference stats."""
    report = []
    try:
        train = import_train(app_dir)
    except Exception:
        return False, [{"error": "import failed", "tb": traceback.format_exc()[-2000:]}]
    ok_all = True
    d_in = teacher["d_in"]
    for chk in spec["checks"]:
        cfg = chk["cfg"]
        xtr, ytr = sample(teacher, chk["n_train"], chk["train_data_seed"])
        xte, yte = sample(teacher, chk["n_test"], chk["test_data_seed"])
        vals = []
        err = None
        for s in chk["seeds"]:
            try:
                with np.errstate(all="ignore"):
                    p = train(dict(cfg), xtr.copy(), ytr.copy(), s)
                shapes = {"W1": (d_in, cfg["hidden"]), "b1": (cfg["hidden"],), "W2": (cfg["hidden"], 1), "b2": (1,)}
                for k, shp in shapes.items():
                    if tuple(np.shape(p[k])) != shp:
                        raise ValueError("bad shape for %s: %s" % (k, np.shape(p[k])))
                v = ref_mse({k: np.asarray(p[k], dtype=float) for k in shapes}, xte, yte)
                if not np.isfinite(v):
                    raise FloatingPointError("non-finite mse")
                vals.append(v)
            except Exception as e:
                err = "%s: %s" % (type(e).__name__, str(e)[:200])
                break
        if err:
            ok = False
            rec = {"name": chk["name"], "error": err}
        else:
            m = float(np.mean(vals))
            ok = abs(m - chk["ref_mean"]) <= chk["tol"]
            rec = {"name": chk["name"], "mean": round(m, 6), "ref_mean": round(chk["ref_mean"], 6),
                   "tol": round(chk["tol"], 6), "pass": bool(ok)}
        ok_all &= ok
        report.append(rec)
        if verbose:
            print(json.dumps(rec))
    return bool(ok_all), report
