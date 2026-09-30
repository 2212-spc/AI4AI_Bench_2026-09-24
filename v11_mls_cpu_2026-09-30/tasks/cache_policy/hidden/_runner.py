"""Child side of the lock-step cache protocol (copied next to the agent's solution by grade.py).

Host -> child, one line at a time:
    "C <capacity>"        once, first line
    "H <t> <key>"         access t is a hit
    "M <t> <key>"         access t is a miss
Child -> host, exactly one line per H/M line:
    ""                    (hit ack, or: no victim / bypass)
    "<victim key>"        on a miss: evict this resident key
The host sends access t+1 only after it has read the reply to access t, so the policy never sees the
future.  The solution's own prints go to stderr (the protocol uses a private copy of fd 1).
"""
import importlib.util, os, sys

sol_path = sys.argv[1]
proto_out = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)                                   # anything the solution prints goes to stderr
sys.stdout = sys.stderr
os.chdir(os.path.dirname(sol_path)); sys.path.insert(0, os.path.dirname(sol_path))
spec = importlib.util.spec_from_file_location("solution", sol_path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

inp = sys.stdin
first = inp.readline().split()
pol = mod.Policy(int(first[1]))
proto_out.write("ready\n")
for line in inp:
    tag, t, key = line.rstrip("\n").split(" ", 2)
    if tag == "H":
        pol.on_hit(key, int(t))
        proto_out.write("\n")
    else:
        v = pol.on_miss(key, int(t))
        proto_out.write(("" if v is None else str(v)) + "\n")
