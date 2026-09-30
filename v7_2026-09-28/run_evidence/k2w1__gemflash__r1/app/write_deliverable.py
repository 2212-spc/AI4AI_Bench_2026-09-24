import json

output = {
    "mixture": {
        "web": 0.57,
        "code": 0.32,
        "math": 0.02,
        "papers": 0.09
    },
    "predicted_loss": {
        "lo": 1.413,
        "hi": 1.461
    }
}

# Verify validity:
mix = output["mixture"]
weights = list(mix.values())
assert all(w >= 0 for w in weights), "Negative weights!"
assert abs(sum(weights) - 1.0) < 1e-9, f"Sum != 1: {sum(weights)}"
assert set(mix.keys()) == {"web", "code", "math", "papers"}, f"Keys mismatch: {mix.keys()}"

pred = output["predicted_loss"]
lo = pred["lo"]
hi = pred["hi"]
width = hi - lo
assert lo < hi, "lo >= hi"
assert width <= 0.05 + 1e-9, f"Width {width} exceeds 0.05"

with open("/app/mixture.json", "w") as f:
    json.dump(output, f, indent=2)

print("mixture.json successfully written and validated:")
print(json.dumps(output, indent=2))
print(f"Interval width: {width:.4f}")
