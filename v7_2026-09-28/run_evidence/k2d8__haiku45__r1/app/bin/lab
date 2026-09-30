#!/usr/bin/env python3
"""`lab` - client for the simulated lab.

  lab spec                      what the lab offers, costs, your remaining budget
  lab status                    budget spent / left
  lab history [last=20]         your recent calls and their results
  lab <op> key=value ...        run an operation (see `lab spec`)
  lab <op> --json '{...}'       same, arguments as one JSON object (use for lists / nested values)
  lab <op> --file args.json     same, arguments read from a JSON file

Every successful call is also appended to /app/lab_log.jsonl.  Results are JSON on stdout."""
import json, os, sys, urllib.request

URL = os.environ.get("LAB_URL", "http://127.0.0.1:18787") + "/api"
APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_val(v):
    try:
        return json.loads(v)
    except Exception:
        return v


def call(op, args):
    tok = open(os.path.join(APP, ".lab_token")).read().strip()
    body = json.dumps({"token": tok, "op": op, "args": args}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=170) as r:
        return json.loads(r.read())


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__); return 0
    op, rest = argv[0], argv[1:]
    args = {}
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--json":
            args.update(json.loads(rest[i + 1])); i += 2; continue
        if a == "--file":
            args.update(json.load(open(rest[i + 1]))); i += 2; continue
        if "=" not in a:
            print(json.dumps({"ok": False, "error": "arguments must be key=value (got %r)" % a})); return 2
        k, v = a.split("=", 1)
        args[k] = parse_val(v); i += 1
    try:
        out = call(op, args)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "lab unreachable: %s" % type(e).__name__})); return 3
    if out.get("ok") and op not in ("spec", "status", "history"):
        # CONVENIENCE LOG ONLY - NOT THE LEDGER.  This records calls that went through this CLI; an
        # agent that calls the lab programmatically (most do, eventually) bypasses it entirely.  The
        # authoritative record is <run_dir>/lab_ledger.jsonl, written server-side, and grade.json's
        # `lab_calls` is counted from that.
        #
        # This burned us on 2026-09-28: k2w1__gemflash__r1 shows 4 calls here and 69 in the ledger
        # (29 distinct mixtures, 9 distinct (N,D) points), and an analysis built on this file
        # concluded the model had passed by reciting a published mixture instead of experimenting.
        # It had not.  Seven gemflash runs are under-counted here, one by 366x (733 real vs 2 logged).
        with open(os.path.join(APP, "lab_log.jsonl"), "a") as f:
            f.write(json.dumps({"op": op, "args": args, "result": out["result"],
                                "_note": "CLI calls only; authoritative record is lab_ledger.jsonl"}) + "\n")
    print(json.dumps(out.get("result") if out.get("ok") else out, indent=1))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
