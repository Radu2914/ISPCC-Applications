import pandas as pd
from intentional_doe_publish_v2 import Surrogate4SIMM

BOUNDARY_THRESHOLD = 0.4
MAX_NEIGHBOUR_DIST = 0.5

model = Surrogate4SIMM.load("surrogate_4simm_last.joblib")

print("\nEnter design point values:")
gap   = float(input("  gap                        : "))
upper = float(input("  upper_protective_layer     : "))
lower = float(input("  lower_protective_layer     : "))
diel  = float(input("  protective_layer_dielectric: "))

# ── Input validation ────────────────────────────────────────
errors = []
if gap <= 0:
    errors.append("  gap must be > 0  (physically: module cannot touch phantom)")
if upper < 0:
    errors.append("  upper_protective_layer must be >= 0")
if lower < 0:
    errors.append("  lower_protective_layer must be >= 0")
if not (0.001 <= diel <= 1.0):
    errors.append("  protective_layer_dielectric must be between 0.001 and 1.0")
if upper + lower == 0:
    errors.append("  total protective layer thickness cannot be zero")

if errors:
    print("\n  !! INVALID INPUT ──────────────────────────────────────")
    for e in errors:
        print(e)
    print("  Prediction aborted.")
    raise SystemExit

df_new = pd.DataFrame([{
    "gap":                         gap,
    "upper_protective_layer":      upper,
    "lower_protective_layer":      lower,
    "protective_layer_dielectric": diel,
}])

results = model.predict_with_diagnostics(df_new, boundary_threshold=BOUNDARY_THRESHOLD)
r = results[0]

print()
print("  ── Prediction ─────────────────────────────────────────")
print(f"  Predicted SAR        : {r['prediction']:.4f} W/m²")
print(f"  Local error (±)      : {r['local_mae']:.4f} W/m²"
      "  ← avg surrogate error at 2 nearest HFSS points")
print()
print("  ── Local neighbourhood ────────────────────────────────")
print(f"  Nearest HFSS SAR     : {r['nearest_sars'][0]:.4f} W/m²"
      f"  (dist={r['nearest_dists'][0]:.4f})")
print(f"  2nd nearest HFSS SAR : {r['nearest_sars'][1]:.4f} W/m²"
      f"  (dist={r['nearest_dists'][1]:.4f})")
print(f"  Neighbour spread     : {r['neighbour_spread']:.4f} W/m²"
      f"  (threshold: {BOUNDARY_THRESHOLD})")
print()

if r["nearest_dists"][0] > MAX_NEIGHBOUR_DIST:
    print("  !! OUT OF DISTRIBUTION ────────────────────────────────")
    print(f"  Nearest HFSS point is {r['nearest_dists'][0]:.4f} away")
    print(f"  (threshold: {MAX_NEIGHBOUR_DIST}). This input combination")
    print("  was never simulated. Prediction is pure extrapolation.")
    print("  Do not use this result under any circumstances.")
elif r["boundary_warning"]:
    print("  !! BOUNDARY ZONE ─────────────────────────────────────")
    print(f"  The two nearest HFSS simulations differ by"
          f" {r['neighbour_spread']:.4f} W/m²,")
    print(f"  which exceeds the {BOUNDARY_THRESHOLD} W/m² threshold.")
    print("  This design sits in a high-gradient region of the space.")
    print("  Do not use this prediction for a compliance decision")
    print("  without running a full HFSS simulation.")
else:
    print("  OK — Stable region ────────────────────────────────────")
    print("  Neighbour spread is within threshold.")
    print("  Prediction is locally consistent with HFSS data.")