import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = r"C:\Users\Victus\OneDrive\Desktop\AQI\data\processed\processed_dataset.csv"
MODEL_DIR  = r"C:\Users\Victus\OneDrive\Desktop\AQI\models"
OUTPUT_DIR = r"C:\Users\Victus\OneDrive\Desktop\AQI\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print("=== Loading model ===")
model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "cnn_lstm_best.h5"),
    compile=False
)
model.compile(optimizer="adam", loss="huber", metrics=["mae"])
scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))
print("  Loaded")

# ── Features — must match train_model.py ─────────────────────────────────────
FEATURES = [
    "NO2_sat", "SO2_sat", "CO_sat", "O3_sat", "HCHO_sat",
    "NO2", "CO", "O3", "SO2", "temp", "rh", "wind_speed",
    "lat", "lon",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "dayofweek", "season",
    "PM25_lag1", "PM25_lag3", "PM25_lag7",
    "PM25_roll3", "PM25_roll7", "AQI_lag1",
]
TARGET = "AQI"

# ── Load and predict ──────────────────────────────────────────────────────────
print("\n=== Running predictions on full dataset ===")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

df_model = df.dropna(subset=FEATURES + [TARGET]).copy()
X = scaler_X.transform(df_model[FEATURES].values)
X = X.reshape(X.shape[0], 1, X.shape[1])

y_pred_scaled = model.predict(X, verbose=0).flatten()
y_pred = scaler_y.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).flatten()
y_true = df_model[TARGET].values

df_model["predicted_AQI"] = y_pred
df_model["error"]         = y_true - y_pred
df_model["abs_error"]     = np.abs(y_true - y_pred)

# ── Overall metrics ───────────────────────────────────────────────────────────
print("\n=== Overall Model Accuracy ===")
rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
mae   = mean_absolute_error(y_true, y_pred)
r2    = r2_score(y_true, y_pred)
bias  = np.mean(y_pred - y_true)
corr  = np.corrcoef(y_true, y_pred)[0, 1]

print(f"  RMSE        : {rmse:.2f}")
print(f"  MAE         : {mae:.2f}")
print(f"  R²          : {r2:.4f}")
print(f"  Correlation : {corr:.4f}")
print(f"  Bias        : {bias:.2f}")

# ── Per station metrics ───────────────────────────────────────────────────────
print("\n=== Per Station Accuracy ===")
station_rows = []
for station in sorted(df_model["station"].unique()):
    s = df_model[df_model["station"] == station]
    if len(s) < 5:
        continue
    sr   = np.sqrt(mean_squared_error(s[TARGET], s["predicted_AQI"]))
    sm   = mean_absolute_error(s[TARGET], s["predicted_AQI"])
    sr2  = r2_score(s[TARGET], s["predicted_AQI"])
    sbias = (s["predicted_AQI"] - s[TARGET]).mean()
    station_rows.append({
        "Station": station,
        "State":   s["state"].iloc[0],
        "N":       len(s),
        "RMSE":    round(sr, 2),
        "MAE":     round(sm, 2),
        "R2":      round(sr2, 4),
        "Bias":    round(sbias, 2),
    })
    print(f"  {station:20s}  RMSE={sr:.1f}  MAE={sm:.1f}  R²={sr2:.3f}  Bias={sbias:.1f}")

station_df = pd.DataFrame(station_rows)
station_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation_per_station.csv"), index=False
)

# ── Season metrics ────────────────────────────────────────────────────────────
print("\n=== Season-wise Accuracy ===")
season_map  = {0: "Winter", 1: "Summer", 2: "Monsoon", 3: "Post-Monsoon"}
season_rows = []
for sid, sname in season_map.items():
    s = df_model[df_model["season"] == sid]
    if len(s) < 5:
        continue
    sr  = np.sqrt(mean_squared_error(s[TARGET], s["predicted_AQI"]))
    sm  = mean_absolute_error(s[TARGET], s["predicted_AQI"])
    sr2 = r2_score(s[TARGET], s["predicted_AQI"])
    season_rows.append({
        "Season": sname, "N": len(s),
        "RMSE": round(sr,2), "MAE": round(sm,2), "R2": round(sr2,4)
    })
    print(f"  {sname:15s}  N={len(s):4d}  RMSE={sr:.1f}  MAE={sm:.1f}  R²={sr2:.3f}")

season_df = pd.DataFrame(season_rows)
season_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation_per_season.csv"), index=False
)

# ── AQI category accuracy ─────────────────────────────────────────────────────
print("\n=== AQI Category Accuracy ===")

def aqi_category(aqi):
    if aqi <= 50:   return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else:            return "Severe"

df_model["actual_cat"]    = df_model[TARGET].apply(aqi_category)
df_model["predicted_cat"] = df_model["predicted_AQI"].apply(aqi_category)
df_model["cat_correct"]   = (
    df_model["actual_cat"] == df_model["predicted_cat"]
)
cat_accuracy = df_model["cat_correct"].mean() * 100
print(f"  Category match accuracy: {cat_accuracy:.1f}%")

# ── Save full results ─────────────────────────────────────────────────────────
summary = {
    "Metric": ["RMSE", "MAE", "R2", "Correlation", "Bias",
               "Category_Accuracy_%", "Total_Samples", "N_Stations"],
    "Value":  [round(rmse,2), round(mae,2), round(r2,4),
               round(corr,4), round(bias,2),
               round(cat_accuracy,1), len(df_model),
               df_model["station"].nunique()]
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation_summary.csv"), index=False
)
print("\n  Saved → outputs/validation_summary.csv")

# ════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════

# ── Plot 1: Summary dashboard ─────────────────────────────────────────────────
print("\n=== Generating plots ===")

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1a. Actual vs Predicted scatter
ax1 = fig.add_subplot(gs[0, :2])
scatter = ax1.scatter(y_true, y_pred, c=y_true,
                       cmap="RdYlGn_r", alpha=0.4, s=12)
mn, mx = 0, 500
ax1.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect fit")
ax1.set_xlim(mn, mx)
ax1.set_ylim(mn, mx)
ax1.set_xlabel("Actual AQI",    fontsize=11)
ax1.set_ylabel("Predicted AQI", fontsize=11)
ax1.set_title(f"Actual vs Predicted AQI — All Stations\n"
              f"RMSE={rmse:.1f} | MAE={mae:.1f} | R²={r2:.3f}",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label="Actual AQI")

# 1b. Error distribution
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(df_model["error"], bins=40, color="steelblue",
          edgecolor="white", alpha=0.8)
ax2.axvline(0, color="red", lw=2, linestyle="--")
ax2.axvline(bias, color="orange", lw=1.5,
             linestyle="--", label=f"Bias={bias:.1f}")
ax2.set_xlabel("Error (Actual - Predicted)", fontsize=10)
ax2.set_ylabel("Count", fontsize=10)
ax2.set_title("Prediction Error Distribution", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 1c. Per station R² bar chart
ax3 = fig.add_subplot(gs[1, :])
colors_bar = ["#2ecc71" if r >= 0.85 else
              "#f39c12" if r >= 0.7 else
              "#e74c3c" for r in station_df["R2"]]
bars = ax3.bar(station_df["Station"], station_df["R2"],
               color=colors_bar, edgecolor="white", alpha=0.85)
ax3.axhline(0.85, color="green",  lw=1.5,
             linestyle="--", label="R²=0.85 (good)")
ax3.axhline(0.70, color="orange", lw=1.5,
             linestyle="--", label="R²=0.70 (acceptable)")
ax3.set_ylim(0, 1.05)
ax3.set_xlabel("Station", fontsize=11)
ax3.set_ylabel("R²",      fontsize=11)
ax3.set_title("R² per Station", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis="y")
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
for bar, val in zip(bars, station_df["R2"]):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8)

# 1d. Season RMSE bar
ax4 = fig.add_subplot(gs[2, 0])
if len(season_df) > 0:
    ax4.bar(season_df["Season"], season_df["RMSE"],
            color=["#3498db","#e74c3c","#27ae60","#f39c12"],
            edgecolor="white", alpha=0.85)
    ax4.set_ylabel("RMSE", fontsize=10)
    ax4.set_title("RMSE by Season", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=20)

# 1e. Station RMSE bar
ax5 = fig.add_subplot(gs[2, 1])
ax5.barh(station_df["Station"], station_df["RMSE"],
          color="salmon", edgecolor="white", alpha=0.85)
ax5.set_xlabel("RMSE", fontsize=10)
ax5.set_title("RMSE per Station", fontsize=11, fontweight="bold")
ax5.grid(True, alpha=0.3, axis="x")

# 1f. Metrics table
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis("off")
table_data = [
    ["Metric", "Value"],
    ["RMSE",   f"{rmse:.2f}"],
    ["MAE",    f"{mae:.2f}"],
    ["R²",     f"{r2:.4f}"],
    ["Corr",   f"{corr:.4f}"],
    ["Bias",   f"{bias:.2f}"],
    ["Cat Acc",f"{cat_accuracy:.1f}%"],
    ["Samples",f"{len(df_model)}"],
]
table = ax6.table(cellText=table_data[1:],
                   colLabels=table_data[0],
                   cellLoc="center", loc="center",
                   bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#ecf0f1")
ax6.set_title("Model Summary", fontsize=11, fontweight="bold")

plt.suptitle("CNN-LSTM AQI Model — Validation Dashboard\n"
             "Satellite-based Surface AQI Prediction over India (2025)",
             fontsize=14, fontweight="bold", y=1.01)

plt.savefig(os.path.join(OUTPUT_DIR, "validation_dashboard.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  Saved → outputs/validation_dashboard.png")

# ── Plot 2: Time series all stations ─────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()
stations = sorted(df_model["station"].unique())

for i, station in enumerate(stations[:9]):
    s = df_model[df_model["station"] == station].sort_values("date")
    ax = axes[i]
    ax.plot(s["date"], s[TARGET],
            label="Actual", color="red", lw=1.2, alpha=0.8)
    ax.plot(s["date"], s["predicted_AQI"],
            label="Predicted", color="blue",
            lw=1.2, linestyle="--", alpha=0.8)
    ax.fill_between(s["date"], s[TARGET], s["predicted_AQI"],
                    alpha=0.1, color="purple")
    sr2 = r2_score(s[TARGET], s["predicted_AQI"])
    ax.set_title(f"{station} — R²={sr2:.3f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Date", fontsize=7)
    ax.set_ylabel("AQI",  fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=6)

for j in range(len(stations), 9):
    axes[j].set_visible(False)

plt.suptitle("AQI Time Series — Actual vs Predicted (All Stations 2025)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "timeseries_all_stations.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("  Saved → outputs/timeseries_all_stations.png")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  FINAL VALIDATION SUMMARY — OBJECTIVE 1")
print("="*55)
print(f"  RMSE             : {rmse:.2f}")
print(f"  MAE              : {mae:.2f}")
print(f"  R²               : {r2:.4f}")
print(f"  Correlation      : {corr:.4f}")
print(f"  Bias             : {bias:.2f}")
print(f"  Category Accuracy: {cat_accuracy:.1f}%")
print(f"  Total samples    : {len(df_model)}")
print(f"  Stations         : {df_model['station'].nunique()}")
print("="*55)
print("\nOutput files:")
print("  outputs/validation_dashboard.png")
print("  outputs/timeseries_all_stations.png")
print("  outputs/validation_summary.csv")
print("  outputs/validation_per_station.csv")
print("  outputs/validation_per_season.csv")
print("\n=== Objective 1 Complete ===")