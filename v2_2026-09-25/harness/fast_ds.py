"""gen_ds/gen_do with a new world seed, and the per-reading answer sweep in select() spread over 4 processes.
Pure speed patch: the answers are the same function of the same inputs, only computed in parallel."""
import runpy, sys, os, time, multiprocessing as mp
gen, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
here = os.path.dirname(gen)
sys.path.insert(0, here); sys.path.insert(0, os.path.join(here, "..", "core"))
import world_d
orig_build = world_d.build
world_d.build = lambda s=0: orig_build(seed)
import gen_ds as GS
_B = _Q = None
def _work(name):
    d = dict(_B.each())[name]
    return name, _B.answer(d, _Q)
orig_select = GS.select
def select(b, items, plans, seeds=()):
    global _B, _Q
    qall = {k: [dict(q, id="x%03d" % i) for i, q in enumerate(items[k])] for k in items}
    qall["plans"] = [dict(q, id="x%03d" % i) for i, q in enumerate(plans)]
    _B, _Q = b, qall
    t = time.time()
    with mp.get_context("fork").Pool(4) as p:
        cache = dict(p.map(_work, [n for n, _ in b.each()]))
    print("parallel sweep %.1fs over %d readings" % (time.time() - t, len(cache)), flush=True)
    ids = {id(d): n for n, d in b.each()}
    real = b.answer
    def answer(d, q):
        if q == qall and id(d) in ids:
            return cache[ids[id(d)]]
        return real(d, q)
    b.answer = answer
    try:
        return orig_select(b, items, plans, seeds)
    finally:
        b.answer = real
GS.select = select
sys.argv = [gen, out]
t0 = time.time()
try:
    if gen.endswith("gen_ds.py"):
        GS.OUT = out
        rc = GS.main()
    else:
        runpy.run_path(gen, run_name="__main__")
        rc = 0
finally:
    print("wall %.1fs" % (time.time() - t0), flush=True)
