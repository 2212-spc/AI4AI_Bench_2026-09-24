#!/usr/bin/env python3
"""lab - client for the ScaleLab training service.

  lab spec                                   knobs, fixed settings, caps, metrics (free)
  lab run N=6e7 D=1.5e10 lr=0.002 seed=3     one run; prints JSON (list knobs as key=value; lists as a,b,c)
  lab batch plan.json                        run a JSON list of requests in order; stops at the first refusal
  lab status                                 compute used / left (free)
  lab history                                every run so far (free)

Every successful run is also appended to /app/lab_runs.jsonl."""
import json, os, sys, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
URL = os.environ.get("LAB_URL", "http://127.0.0.1:18777/")


def token():
    for p in (os.path.join(APP, ".lab_token"), os.path.join(HERE, ".lab_token")):
        if os.path.exists(p):
            return open(p).read().strip()
    sys.exit("lab: no .lab_token found (is the lab service configured for this task?)")


def call(req):
    req["token"] = token()
    r = urllib.request.Request(URL, data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # the lab is local; never use a proxy
    try:
        with opener.open(r, timeout=120) as f:
            return json.loads(f.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read() or b"{}")
        except Exception:
            return {"error": "lab HTTP %d" % e.code}
    except Exception as e:
        return {"error": "lab unreachable: %r" % (e,)}


def parse_val(v):
    if "," in v:
        return [parse_val(x) for x in v.split(",") if x != ""]
    try:
        f = float(v)
        return int(f) if f.is_integer() and "e" not in v.lower() and "." not in v else f
    except ValueError:
        return v


def record(results):
    try:
        with open(os.path.join(APP, "lab_runs.jsonl"), "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    except Exception:
        pass


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help", "help"):
        print(__doc__); return 0
    cmd = a[0]
    if cmd == "run":
        req = {}
        for kv in a[1:]:
            if "=" not in kv:
                print(json.dumps({"error": "arguments must be key=value, got %r" % kv})); return 2
            k, v = kv.split("=", 1); req[k.strip()] = parse_val(v.strip())
        out = call({"op": "run", "req": req})
        if "error" not in out:
            record([out])
        print(json.dumps(out)); return 1 if "error" in out else 0
    if cmd == "batch":
        if len(a) < 2:
            print(json.dumps({"error": "usage: lab batch plan.json"})); return 2
        try:
            reqs = json.load(open(a[1]))
            assert isinstance(reqs, list)
        except Exception as e:
            print(json.dumps({"error": "plan must be a JSON list of run requests: %r" % (e,)})); return 2
        out = call({"op": "batch", "reqs": reqs})
        record(out.get("results", []))
        print(json.dumps(out)); return 1 if out.get("error") else 0
    if cmd in ("spec", "status", "history"):
        out = call({"op": cmd}); print(json.dumps(out, indent=1)); return 1 if "error" in out else 0
    print(json.dumps({"error": "unknown command %r (spec | run | batch | status | history)" % cmd})); return 2


if __name__ == "__main__":
    sys.exit(main())
