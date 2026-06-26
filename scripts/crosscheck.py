import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_squared_error

OUTPUT_DIR = r"C:\Users\Victus\OneDrive\Desktop\AQI\outputs"
DATA_DIR   = r"C:\Users\Victus\OneDrive\Desktop\AQI\data\processed"

# Load validation results
val = pd.read_csv(os.path.join(OUTPUT_DIR, "validation_results.csv"))
df  = pd.read_csv(os.path.join(DATA_DIR,   "processed_dataset.csv"))

print("=" * 55)
print("CROSS-CHECK REPORT")
print("=" * 55)

# ── Check 1: Are predictions in valid AQI range? ──────────────────────────────
print("\n[Check 1] Prediction range validity")
print(f"  Actual AQI   → min: {val['actual_AQI'].min():.1f}  "
      f"max: {val['actual_AQI'].max():.1f}  "
      f"mean: {val['actual_AQI'].mean():.1f}")
print(f"  Predicted AQI → min: {val['predicted_AQI'].min():.1f}  "
      f"max: {val['predicted_AQI'].max():.1f}  "
      f"mean: {val['predicted_AQI'].mean():.1f}")
out_of_range = ((val["predicted_AQI"] < 0) |
                (val["predicted_AQI"] > 500)).sum()
print(f"  Out of range predictions (0-500): {out_of_range}")
if out_of_range == 0:
    print("  ✓ All predictions in valid range")
else:
    print("  ✗ Some predictions out of range — check model")

# ── Check 2: Is R² genuinely good? ───────────────────────────────────────────
print("\n[Check 2] R² sanity check")
r2 = r2_score(val["actual_AQI"], val["predicted_AQI"])
rmse = np.sqrt(mean_squared_error(val["actual_AQI"], val["predicted_AQI"]))
print(f"  R²   = {r2:.4f}")
print(f"  RMSE = {rmse:.2f}")
if r2 > 0.95:
    print("  ⚠ R² > 0.95 — possible overfitting, check train/test split")
elif r2 > 0.85:
    print("  ✓ R² is excellent — model is reliable")
elif r2 > 0.70:
    print("  ~ R² is acceptable — model is usable")
else:
    print("  ✗ R² too low — model needs improvement")

# ── Check 3: Per station check ────────────────────────────────────────────────
print("\n[Check 3] Per station mean AQI comparison")
print(f"  {'Station':<20} {'Actual Mean':>12} {'Predicted Mean':>15} {'Diff':>8}")
print(f"  {'-'*58}")
for station in sorted(val["station"].unique()):
    s = val[val["station"] == station]
    actual_mean = s["actual_AQI"].mean()
    pred_mean   = s["predicted_AQI"].mean()
    diff        = pred_mean - actual_mean
    flag = "✓" if abs(diff) < 30 else "⚠"
    print(f"  {station:<20} {actual_mean:>12.1f} "
          f"{pred_mean:>15.1f} {diff:>+8.1f} {flag}")

# ── Check 4: Error distribution ───────────────────────────────────────────────
print("\n[Check 4] Error distribution")
errors = val["actual_AQI"] - val["predicted_AQI"]
within_20  = (np.abs(errors) <= 20).mean()  * 100
within_50  = (np.abs(errors) <= 50).mean()  * 100
within_100 = (np.abs(errors) <= 100).mean() * 100
print(f"  Predictions within ±20  AQI: {within_20:.1f}%")
print(f"  Predictions within ±50  AQI: {within_50:.1f}%")
print(f"  Predictions within ±100 AQI: {within_100:.1f}%")
if within_50 > 80:
    print("  ✓ Good — 80%+ predictions within ±50 AQI")
else:
    print("  ⚠ Below 80% within ±50 — model may need tuning")

# ── Check 5: Known city AQI reality check ────────────────────────────────────
print("\n[Check 5] Reality check vs known India AQI ranges")
known_ranges = {
    "Narela_Delhi": (150, 450),
    "Amritsar":     (80,  300),
    "Lucknow":      (100, 350),
    "Mumbai":       (50,  180),
    "Bengaluru":    (30,  120),
    "Kolkata":      (80,  300),
    "Jaipur":       (60,  250),
    "Gurugram":     (100, 400),
    "Chandigarh":   (80,  280),
}
for station, (lo, hi) in known_ranges.items():
    s = val[val["station"] == station]
    if len(s) == 0:
        continue
    pred_mean = s["predicted_AQI"].mean()
    if lo <= pred_mean <= hi:
        print(f"  ✓ {station:<20} predicted mean {pred_mean:.0f} "
              f"(expected {lo}-{hi})")
    else:
        print(f"  ✗ {station:<20} predicted mean {pred_mean:.0f} "
              f"(expected {lo}-{hi}) — CHECK THIS")

# ── Plot: Actual vs Predicted with perfect reference ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter
axes[0].scatter(val["actual_AQI"], val["predicted_AQI"],
                alpha=0.4, s=12, color="steelblue")
axes[0].plot([0, 500], [0, 500], "r--", lw=2, label="Perfect fit")
axes[0].plot([0, 500], [50, 550], "g--",
             lw=1, alpha=0.5, label="+50 error band")
axes[0].plot([0, 500], [-50, 450], "g--", lw=1, alpha=0.5)
axes[0].set_xlim(0, 500)
axes[0].set_ylim(0, 500)
axes[0].set_xlabel("Actual AQI")
axes[0].set_ylabel("Predicted AQI")
axes[0].set_title(f"Scatter — R²={r2:.3f} | RMSE={rmse:.1f}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Error histogram
axes[1].hist(errors, bins=40, color="steelblue",
             edgecolor="white", alpha=0.8)
axes[1].axvline(0,            color="red",    lw=2, label="Zero error")
axes[1].axvline( 50,          color="orange", lw=1.5,
                 linestyle="--", label="±50 band")
axes[1].axvline(-50,          color="orange", lw=1.5, linestyle="--")
axes[1].set_xlabel("Prediction Error (Actual - Predicted)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Error Distribution\n"
                  f"{within_50:.1f}% within ±50 AQI")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Cross-Check Report — AQI Model Accuracy",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "crosscheck_report.png"), dpi=150)
plt.show()

print("\n" + "="*55)
print("  Cross-check complete → outputs/crosscheck_report.png")
print("="*55)