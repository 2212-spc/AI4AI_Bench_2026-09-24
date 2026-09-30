"""Host-side grader for cache_policy.  python3 grade.py <agent_workdir>  -> JSON on stdout.

For every setting the host generates the trace, runs the agent's Policy in a child process through a
lock-step pipe (one access at a time, the next key is sent only after the reply to the current one), keeps
the authoritative cache contents itself, and scores
    metric = (miss_LRU - miss_policy) / (miss_LRU - miss_OPT)   on accesses after warm-up
(OPT = Belady MIN with bypass, computed host-side).  Pass = metric >= THRESHOLD on EVERY setting.
"""
import json, os, shutil, subprocess, sys, tempfile, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hidden"))
import numpy as np
import world as W

SOL_FILE = "solution.py"
THRESHOLD = 0.35          # fraction of the LRU->OPT miss gap closed, required on every setting
BUDGET = 150.0            # wall seconds per setting for the child (the reference needs ~3 s)


def run_policy(sol_dir, keys, C, budget=BUDGET):
    """Return (miss bool array, seconds, error or None)."""
    tmp = tempfile.mkdtemp(prefix="v11cache_")
    try:
        work = os.path.join(tmp, "w")
        shutil.copytree(sol_dir, work, ignore=shutil.ignore_patterns("__pycache__", "data", "*.npz"))
        shutil.copy(os.path.join(HERE, "hidden", "_runner.py"), os.path.join(tmp, "_runner.py"))
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        err_f = open(os.path.join(tmp, "stderr.txt"), "w+")
        p = subprocess.Popen([sys.executable, os.path.join(tmp, "_runner.py"), os.path.join(work, SOL_FILE)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err_f, cwd=work,
                             text=True, bufsize=1, env=env)
        timer = threading.Timer(budget, p.kill); timer.start()
        t0 = time.time()
        miss = np.zeros(len(keys), bool)
        res, err = set(), None
        try:
            p.stdin.write(f"C {C}\n"); p.stdin.flush()
            if p.stdout.readline() != "ready\n":
                raise RuntimeError("policy failed to start")
            w, r = p.stdin, p.stdout
            for i, k in enumerate(keys):
                if k in res:
                    w.write(f"H {i} {k}\n"); w.flush()
                    if r.readline() == "":
                        raise RuntimeError(f"policy died at access {i}")
                    continue
                miss[i] = True
                w.write(f"M {i} {k}\n"); w.flush()
                ans = r.readline()
                if ans == "":
                    raise RuntimeError(f"policy died at access {i}")
                d = ans[:-1]
                if len(res) < C:
                    res.add(k)
                elif d:
                    if d not in res:
                        raise RuntimeError(f"access {i}: victim {d!r} is not resident")
                    res.remove(d); res.add(k)
        except (RuntimeError, BrokenPipeError, OSError, ValueError) as e:
            err = str(e)
        finally:
            timer.cancel()
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        secs = time.time() - t0
        if err and secs >= budget - 1:
            err = f"time budget {budget:.0f}s exceeded ({err})"
        if err:
            err_f.seek(0); tail = err_f.read()[-1500:]
            err = err + ("\n" + tail if tail else "")
        err_f.close()
        return miss, secs, err
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def grade(sol_dir):
    settings, ok_all = {}, True
    for row in W.SETTINGS:
        s = W.build_setting(row)
        keys, C, w = s["keys"], s["C"], s["warm"]
        m_lru = W.lru_misses(keys, C)[w:].mean()
        m_opt = W.opt_misses(keys, C)[w:].mean()
        miss, secs, err = run_policy(sol_dir, keys, C)
        if err is None:
            m_pol = float(miss[w:].mean())
            metric = float((m_lru - m_pol) / (m_lru - m_opt))
        else:
            m_pol, metric = float("nan"), -99.0
        ok = metric >= THRESHOLD
        ok_all &= ok
        settings[s["name"]] = dict(metric=metric, threshold=THRESHOLD, ok=bool(ok), hidden=s["hidden"],
                                   secs=round(secs, 1), capacity=C, miss_lru=round(float(m_lru), 4),
                                   miss_opt=round(float(m_opt), 4), miss_policy=round(m_pol, 4),
                                   **({"error": err[:2000]} if err else {}))
    return {"pass": bool(ok_all), "settings": settings}


if __name__ == "__main__":
    print(json.dumps(grade(os.path.abspath(sys.argv[1])), indent=1))
