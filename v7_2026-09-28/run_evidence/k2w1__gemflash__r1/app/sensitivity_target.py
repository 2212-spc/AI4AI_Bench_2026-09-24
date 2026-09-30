import numpy as np
from optimize_target import evaluate_mix

# Center around best:
best_r = (0.57, 0.32, 0.02, 0.09)
losses, base_comp = evaluate_mix(*best_r)
print(f"Base comp: {base_comp:.5f}")

# Check fine variations in each direction
print("\nVarying math fraction (adjusting web):")
for r_m in [0.005, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]:
    rw = 1.0 - 0.32 - r_m - 0.09
    _, comp = evaluate_mix(rw, 0.32, r_m, 0.09)
    print(f"  math={r_m:.3f} (ep={r_m*500/4:.2f}) -> comp={comp:.5f} (diff={comp-base_comp:+.5f})")

print("\nVarying code fraction (adjusting web):")
for r_c in [0.20, 0.25, 0.28, 0.30, 0.32, 0.34, 0.36, 0.40]:
    rw = 1.0 - r_c - 0.02 - 0.09
    _, comp = evaluate_mix(rw, r_c, 0.02, 0.09)
    print(f"  code={r_c:.3f} (ep={r_c*500/40:.2f}) -> comp={comp:.5f} (diff={comp-base_comp:+.5f})")

print("\nVarying papers fraction (adjusting web):")
for r_p in [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]:
    rw = 1.0 - 0.32 - 0.02 - r_p
    _, comp = evaluate_mix(rw, 0.32, 0.02, r_p)
    print(f"  papers={r_p:.3f} (ep={r_p*500/20:.2f}) -> comp={comp:.5f} (diff={comp-base_comp:+.5f})")

