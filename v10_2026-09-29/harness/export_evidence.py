"""Copy finished runs into <v10>/run_evidence with the redactions the deliverable needs.

  python3 harness/export_evidence.py <run_root> <dest> [run_name ...]

Excluded:  <run>/home/, <run>/sbx_tmp/, <run>/app/.lab_token, gateway.log, anything under a directory named
           .cache / __pycache__, and files > 20 MB (training artefacts such as .npy predictions).
Redacted:  lab.json "token" -> "<redacted>"; every occurrence of any run's lab token in any copied text file ->
           "<redacted-lab-token>" (agents sometimes cat the token file); any sk-... string of >= 16 chars that
           is not the documented placeholder -> "<redacted-sk>".
Afterwards the whole destination tree is scanned again and the script exits non-zero if a real-looking key or
a lab token survives."""
import json, os, re, shutil, sys

SK = re.compile(r"sk-[A-Za-z0-9_-]{16,}")
ALLOWED_SK = ("sk-sandbox-placeholder-not-a-credential", "sk-FAKE-REAL-KEY-0123456789abcdefghijKLMN")

MAX_BYTES = 20 * 1024 * 1024


def _sk_sub(m):
    return m.group(0) if m.group(0) in ALLOWED_SK else "<redacted-sk>"


def main():
    root, dest = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "lab.json")))
    tokens = set()
    for d in os.listdir(root):
        fp = os.path.join(root, d, "lab.json")
        if os.path.isfile(fp):
            tokens.add(json.load(open(fp))["token"])
    n_files = n_red = 0
    for name in names:
        src = os.path.join(root, name)
        for dp, dns, fns in os.walk(src):
            rel = os.path.relpath(dp, src)
            dns[:] = [x for x in dns if x not in (".cache", "__pycache__")
                      and not (rel == "." and x in ("home", "sbx_tmp"))]
            for fn in fns:
                if fn == ".lab_token" or fn.endswith((".npy", ".npz", ".pyc")):
                    continue
                s = os.path.join(dp, fn)
                if os.path.getsize(s) > MAX_BYTES:
                    continue
                t = os.path.join(dest, name, rel, fn)
                os.makedirs(os.path.dirname(t), exist_ok=True)
                raw = open(s, "rb").read()
                try:
                    txt = raw.decode("utf-8")
                except UnicodeDecodeError:
                    open(t, "wb").write(raw); n_files += 1
                    continue
                new = txt
                if rel == "." and fn == "lab.json":
                    j = json.loads(txt); j["token"] = "<redacted>"; new = json.dumps(j)
                for tok in tokens:
                    new = new.replace(tok, "<redacted-lab-token>")
                new = SK.sub(_sk_sub, new)
                if new != txt:
                    n_red += 1
                open(t, "w").write(new); n_files += 1
    bad = []
    for dp, _, fns in os.walk(dest):
        for fn in fns:
            b = open(os.path.join(dp, fn), "rb").read().decode("utf-8", "replace")
            if any(tok in b for tok in tokens) or any(m not in ALLOWED_SK for m in SK.findall(b)):
                bad.append(os.path.join(dp, fn))
    print("copied %d files from %d runs, %d files redacted, residual hits %d" % (n_files, len(names), n_red, len(bad)))
    for b in bad:
        print("  RESIDUAL", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
