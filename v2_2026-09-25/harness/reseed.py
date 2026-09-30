"""Run a family generator whose world seed is hard-coded to 0 with a different seed, by patching build()."""
import runpy, sys, os
gen, mod, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, os.path.dirname(gen))
sys.path.insert(0, os.path.join(os.path.dirname(gen), "..", "core"))
m = __import__(mod)
orig = m.build
m.build = lambda s=0, _o=orig: _o(seed)
sys.argv = [gen, out]
runpy.run_path(gen, run_name="__main__")
