"""Verifier for E1 tasks (runs in the separate verifier container; stdlib only).

Reads the agent artifact /app/decision.json and the baked-in hidden truth /tests/scoring.json
(many-seed mean val MSE of all 64 subsets). Never imports or executes agent-produced code.
Binary reward: 1 iff
  (a) regret of the shipped subset  mu[ship]/mu[opt] - 1 <= delta_rel   and
  (b) |predicted_val_mse - mu[ship]| / mu[ship] <= pred_tol.
env overrides (for local self-checks): DECISION, SCORING, REWARD_DIR"""
import json, math, os, sys

NAMES = ["c1", "c2", "c3", "c4", "c5", "c6"]
DECISION = os.environ.get("DECISION", "/app/decision.json")
SCORING = os.environ.get("SCORING", "/tests/scoring.json")
REWARD_DIR = os.environ.get("REWARD_DIR", "/logs/verifier")


def grade():
    sc = json.load(open(SCORING))
    try:
        d = json.load(open(DECISION))
    except Exception as e:
        return 0, {"error": "decision.json missing or not JSON: %r" % e}
    ship, pred = d.get("ship"), d.get("predicted_val_mse")
    if not isinstance(ship, list) or not all(isinstance(c, str) for c in ship):
        return 0, {"error": "ship must be a list of change names"}
    ship = [c.strip().lower() for c in ship]
    if any(c not in NAMES for c in ship) or len(set(ship)) != len(ship):
        return 0, {"error": "ship must contain distinct names from %s" % NAMES}
    if isinstance(pred, bool) or not isinstance(pred, (int, float)) or not math.isfinite(pred) or pred <= 0:
        return 0, {"error": "predicted_val_mse must be a positive finite number"}
    mask = sum(1 << NAMES.index(c) for c in ship)
    mu = sc["mu"]; opt = sc["opt"]
    regret = mu[mask] / mu[opt] - 1
    perr = abs(pred - mu[mask]) / mu[mask]
    ok = regret <= sc["delta_rel"] and perr <= sc["pred_tol"]
    return int(ok), {"ship": ship, "regret": round(regret, 5), "delta_rel": sc["delta_rel"],
                     "pred_err": round(perr, 5), "pred_tol": sc["pred_tol"], "pass": bool(ok)}


if __name__ == "__main__":
    r, info = grade()
    os.makedirs(REWARD_DIR, exist_ok=True)
    open(os.path.join(REWARD_DIR, "reward.txt"), "w").write("%d\n" % r)
    print(json.dumps(info))
    sys.exit(0)
