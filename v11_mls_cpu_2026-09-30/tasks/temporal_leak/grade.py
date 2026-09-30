"""Host-side grader for temporal_leak.  python3 grade.py <agent_workdir> -> JSON on stdout.

Per setting: the child fits the fixed model (workdir/model.py) on features.build(dump, train_rows); then the
host replays N_LIVE live days in lock-step: for day d it sends the warehouse exactly as of the end of day d-1
and the rows (store, d), and only after the reply moves on.  Truth = final units of day d.
  wape  = sum|pred - y| / sum y,   ratio = wape / wape of an information-fair one-step oracle
  bias  = sum(pred) / sum(y) - 1
Pass on a setting: ratio <= RATIO_MAX and |bias| <= BIAS_MAX.  metric = max(ratio/RATIO_MAX, |bias|/BIAS_MAX),
so metric <= 1 is a pass.  Pass the task = pass on every setting (2 dev + 4 hidden)."""
import json, os, pickle, shutil, struct, subprocess, sys, tempfile, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hidden"))
import numpy as np
import world as W

SOL_FILE = "features.py"
RATIO_MAX, BIAS_MAX, THRESHOLD = 1.55, 0.03, 1.0
BUDGET = 900.0


def _send(p, obj):
    b = pickle.dumps(obj); p.stdin.write(struct.pack("<Q", len(b)) + b); p.stdin.flush()


def _recv(p):
    h = p.stdout.read(8)
    if len(h) < 8: raise RuntimeError("child died")
    r = pickle.loads(p.stdout.read(struct.unpack("<Q", h)[0]))
    if r[0] != "ok": raise RuntimeError(r[1])
    return r[1]


def run_setting(sol_dir, s):
    tmp = tempfile.mkdtemp(prefix="v11leak_")
    try:
        work = os.path.join(tmp, "w")
        shutil.copytree(sol_dir, work, ignore=shutil.ignore_patterns("__pycache__", "data"))
        shutil.copy(os.path.join(HERE, "workdir", "model.py"), os.path.join(work, "model.py"))   # model is fixed
        shutil.copy(os.path.join(HERE, "hidden", "_runner.py"), os.path.join(tmp, "_runner.py"))
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", LOKY_MAX_CPU_COUNT="2")
        err_f = open(os.path.join(tmp, "stderr.txt"), "w+")
        p = subprocess.Popen([sys.executable, os.path.join(tmp, "_runner.py")], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=err_f, cwd=work, env=env)
        timer = threading.Timer(BUDGET, p.kill); timer.start()
        t0 = time.time(); preds, ys, err = [], [], None
        try:
            _send(p, ("train", s["dump"], s["train_rows"])); _recv(p)
            fin = s["final"]
            for d in s["live_days"]:
                rows = fin[fin.day == d][["store", "day", "units"]].sort_values("store").reset_index(drop=True)
                _send(p, ("day", W.as_of(s["tables"], d - 1), rows[["store", "day"]]))
                pr = np.asarray(_recv(p), float)
                if pr.shape != (len(rows),) or not np.all(np.isfinite(pr)): raise RuntimeError(f"bad predictions on day {d}")
                preds.append(np.clip(pr, 0, None)); ys.append(rows.units.to_numpy())
        except (RuntimeError, BrokenPipeError, OSError, ValueError) as e:
            err = str(e)[-1500:]
        finally:
            timer.cancel()
            try: p.stdin.close()
            except Exception: pass
            try: p.wait(timeout=10)
            except Exception: p.kill()
        secs = time.time() - t0
        if err:
            if secs >= BUDGET - 1: err = f"time budget {BUDGET:.0f}s exceeded ({err})"
            err_f.seek(0); err += "\n" + err_f.read()[-1000:]
        err_f.close()
        return (None if err else (np.concatenate(preds), np.concatenate(ys))), secs, err
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def grade(sol_dir):
    settings, ok_all = {}, True
    for row in W.SETTINGS:
        s = W.build_setting(row)
        res, secs, err = run_setting(sol_dir, s)
        if res is None:
            ratio = bias = metric = 99.0
        else:
            p, y = res
            ratio = float(np.abs(p - y).sum() / np.abs(s["oracle"] - y).sum())
            bias = float(p.sum() / y.sum() - 1)
            metric = max(ratio / RATIO_MAX, abs(bias) / BIAS_MAX)
        ok = metric <= THRESHOLD; ok_all &= ok
        settings[s["name"]] = dict(metric=metric, threshold=THRESHOLD, ok=bool(ok), hidden=s["hidden"], secs=round(secs, 1),
                                   wape_ratio=round(ratio, 4), bias=round(bias, 4), **({"error": err} if err else {}))
    return {"pass": bool(ok_all), "settings": settings}


if __name__ == "__main__":
    print(json.dumps(grade(os.path.abspath(sys.argv[1])), indent=1))
