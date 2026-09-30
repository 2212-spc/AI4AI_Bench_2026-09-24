"""usage: launch_e1.py <instance_id> <run_name> <gpt|fable> [hint]"""
import json, os, secrets, subprocess, sys
iid, name, kind = sys.argv[1:4]; hint = len(sys.argv) > 4
root = "/tmp/bench/instances/e1/" + iid; rd = "/tmp/bench/runs/" + name
instr = root + ("/instruction_hint.md" if hint else "/instruction.md")
agent = "gpt_agent.py" if kind == "gpt" else "cc_agent.py"
subprocess.check_call(["python3", "/tmp/bench/harness/" + agent, "init", rd, root + "/public", instr])
tok = secrets.token_hex(12)
open(rd + "/app/.lab_token", "w").write(tok)
json.dump({"token": tok, "instance": iid, "budget": json.load(open(root + "/hidden/scoring.json"))["budget"]}, open(rd + "/lab.json", "w"))
print("launched", rd)
