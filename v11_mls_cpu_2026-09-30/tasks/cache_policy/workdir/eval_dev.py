"""Evaluate a policy on the visible dev traces with the same metric and rules as the grader.

    python3 eval_dev.py [path/to/solution.py]          (default: solution.py)

Per trace:  score = (miss_LRU - miss_policy) / (miss_LRU - miss_OPT)  on accesses after warm-up,
OPT = Belady's offline optimum with bypass.  Pass bar (every setting, dev and hidden): score >= 0.35.
The grader runs the policy in a separate process and feeds it one access at a time; locally we call it
in-process through simulator.simulate (identical cache rules).
"""
import gzip, heapq, importlib.util, os, pickle, sys, time
from collections import OrderedDict
import numpy as np
from simulator import simulate

THRESHOLD = 0.35
HERE = os.path.dirname(os.path.abspath(__file__))


def lru_misses(keys, C):
    c, miss = OrderedDict(), np.zeros(len(keys), bool)
    for i, k in enumerate(keys):
        if k in c:
            c.move_to_end(k)
        else:
            miss[i] = True; c[k] = 1
            if len(c) > C:
                c.popitem(last=False)
    return miss


def opt_misses(keys, C):
    """Belady MIN with bypass (uses the future: only for scoring, never available to a policy)."""
    n = len(keys)
    nxt, last = np.empty(n, np.int64), {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(keys[i], n + i); last[keys[i]] = i
    res, heap, miss = {}, [], np.zeros(n, bool)
    for i, k in enumerate(keys):
        if k in res:
            res[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k)); continue
        miss[i] = True
        if len(res) < C:
            res[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k)); continue
        while True:
            nu, v = heap[0]
            if res.get(v) == -nu: break
            heapq.heappop(heap)
        if nxt[i] >= -nu:
            continue
        heapq.heappop(heap); del res[v]
        res[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k))
    return miss


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "solution.py")
    spec = importlib.util.spec_from_file_location("sol", path)
    sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
    with gzip.open(os.path.join(HERE, "data", "dev_traces.pkl.gz"), "rb") as f:
        traces = pickle.load(f)
    for tr in traces:
        keys, C, w = tr["keys"], tr["capacity"], tr["warm"]
        ml, mo = lru_misses(keys, C)[w:].mean(), opt_misses(keys, C)[w:].mean()
        t = time.time()
        mp = simulate(sol.Policy, keys, C)[w:].mean()
        s = (ml - mp) / (ml - mo)
        print(f"{tr['name']}: C={C} miss LRU={ml:.4f} OPT={mo:.4f} policy={mp:.4f}  "
              f"score={s:+.3f} {'PASS' if s >= THRESHOLD else 'fail'}  ({time.time()-t:.1f}s)")


if __name__ == "__main__":
    main()
