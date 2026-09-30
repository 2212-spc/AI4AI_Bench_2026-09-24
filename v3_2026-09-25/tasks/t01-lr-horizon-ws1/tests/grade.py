"""Grade an answers.json against this task's key.  usage: python3 grade.py answers.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from queries import grade
items = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "key.json")))
try:
    ans = json.load(open(sys.argv[1]))
except Exception as e:
    ans = {}
    print("unreadable answers:", e)
print(json.dumps(grade(items, ans), indent=1))
