import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = r"C:\Users\Victus\OneDrive\Desktop\AQI\data\processed\processed_dataset.csv"
MODEL_DIR  = r"C:\Users\Victus\OneDrive\Desktop\AQI\models"
OUTPUT_DIR = r"C:\Users\Victus\OneDrive\Desktop\AQI\outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("=== Loading data ===")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

# ── Features ──────────────────────────────────────────────────────────────────
FEATURES = [
    # Satellite columns
    "NO2_sat", "SO2_sat", "CO_sat", "O3_sat", "HCHO_sat",
    # Ground measurements
    "NO2", "CO", "O3", "SO2", "temp", "rh", "wind_speed",
    # Location
    "lat", "lon",
    # Temporal
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "dayofweek", "season",
    # Lag features
    "PM25_lag1", "PM25_lag3", "PM25_lag7",
    "PM25_roll3", "PM25_roll7", "AQI_lag1",
]
TARGET = "AQI"

# Keep only rows where all features available
df_model = df.dropna(subset=FEATURES + [TARGET])
print(f"  Rows after dropping NaN: {len(df_model)}")

X = df_model[FEATURES].values
y = df_model[TARGET].values

print(f"  AQI range: {y.min():.1f} to {y.max():.1f}")
print(f"  AQI mean:  {y.mean():.1f}")

# ── Scale ─────────────────────────────────────────────────────────────────────
print("\n=== Scaling features ===")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(MODEL_DIR, "scaler_y.pkl"))
print("  Scalers saved")

# ── Train/test split ──────────────────────────────────────────────────────────
# Use last 20% of dates as test (temporal split — more realistic)
df_model_sorted = df_model.sort_values("date").reset_index(drop=True)
split_idx = int(len(df_model_sorted) * 0.8)

train_idx = df_model_sorted.index[:split_idx]
test_idx  = df_model_sorted.index[split_idx:]

X_all = scaler_X.transform(df_model_sorted[FEATURES].values)
y_all = scaler_y.transform(
    df_model_sorted[TARGET].values.reshape(-1, 1)
).flatten()

X_train = X_all[train_idx].reshape(-1, 1, len(FEATURES))
X_test  = X_all[test_idx].reshape(-1, 1, len(FEATURES))
y_train = y_all[train_idx]
y_test  = y_all[test_idx]

print(f"  Train: {len(X_train)} rows")
print(f"  Test:  {len(X_test)} rows")

# ── Build CNN-LSTM ────────────────────────────────────────────────────────────
print("\n=== Building CNN-LSTM model ===")

def build_model(n_features):
    inp = layers.Input(shape=(1, n_features))

    # CNN branch — spatial feature extraction
    x = layers.Conv1D(128, kernel_size=1, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(256, kernel_size=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    # LSTM branch — temporal learning
    x = layers.LSTM(256, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    # Dense output
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="huber",
        metrics=["mae"]
    )
    return model

model = build_model(len(FEATURES))
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n=== Training model ===")

cb_list = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=20,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=8, min_lr=1e-6, verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "cnn_lstm_best.h5"),
        monitor="val_loss", save_best_only=True, verbose=0
    ),
]

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    callbacks=cb_list,
    verbose=1
)

print("  Training complete")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n=== Evaluating ===")

y_pred_scaled = model.predict(X_test, verbose=0).flatten()
y_pred = scaler_y.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).flatten()
y_true = scaler_y.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

print(f"\n  RMSE : {rmse:.2f}")
print(f"  MAE  : {mae:.2f}")
print(f"  R2   : {r2:.4f}")

# Save results
results_df = pd.DataFrame({
    "date":          df_model_sorted.iloc[test_idx]["date"].values,
    "station":       df_model_sorted.iloc[test_idx]["station"].values,
    "actual_AQI":    y_true,
    "predicted_AQI": y_pred,
    "error":         y_true - y_pred
})
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation_results.csv"), index=False
)
print("  Validation results saved")

# ── Plot 1: Training loss ─────────────────────────────────────────────────────
print("\n=== Generating plots ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["loss"],     label="Train Loss", color="blue")
axes[0].plot(history.history["val_loss"], label="Val Loss",   color="orange")
axes[0].set_title("Training vs Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Huber Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_true, y_pred, alpha=0.4, s=15, color="steelblue")
mn = min(y_true.min(), y_pred.min())
mx = max(y_true.max(), y_pred.max())
axes[1].plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect fit")
axes[1].set_title(f"Actual vs Predicted AQI\nRMSE={rmse:.1f} | R2={r2:.3f}")
axes[1].set_xlabel("Actual AQI")
axes[1].set_ylabel("Predicted AQI")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_validation.png"), dpi=150)
plt.show()
print("  Saved -> outputs/model_validation.png")

# ── Plot 2: Per station accuracy ──────────────────────────────────────────────
stations = results_df["station"].unique()
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.flatten()

for i, station in enumerate(stations[:9]):
    s = results_df[results_df["station"] == station]
    if len(s) < 3:
        continue
    ax = axes[i]
    ax.scatter(s["actual_AQI"], s["predicted_AQI"],
               alpha=0.5, s=20, color="steelblue")
    mn = min(s["actual_AQI"].min(), s["predicted_AQI"].min())
    mx = max(s["actual_AQI"].max(), s["predicted_AQI"].max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
    r2s = r2_score(s["actual_AQI"], s["predicted_AQI"])
    r   = np.sqrt(mean_squared_error(s["actual_AQI"], s["predicted_AQI"]))
    ax.set_title(f"{station}\nR2={r2s:.3f}  RMSE={r:.1f}", fontsize=9)
    ax.set_xlabel("Actual AQI", fontsize=8)
    ax.set_ylabel("Predicted AQI", fontsize=8)
    ax.grid(True, alpha=0.3)

for j in range(len(stations), 9):
    axes[j].set_visible(False)

plt.suptitle("Per-Station Actual vs Predicted AQI",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "validation_per_station.png"), dpi=150)
plt.show()
print("  Saved -> outputs/validation_per_station.png")

# ── Plot 3: Delhi time series ─────────────────────────────────────────────────
delhi = results_df[
    results_df["station"] == "Narela_Delhi"
].sort_values("date")

if len(delhi) > 0:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(delhi["date"], delhi["actual_AQI"],
            label="Actual AQI", color="red", lw=1.5)
    ax.plot(delhi["date"], delhi["predicted_AQI"],
            label="Predicted AQI", color="blue", lw=1.5, linestyle="--")
    ax.fill_between(delhi["date"],
                    delhi["actual_AQI"], delhi["predicted_AQI"],
                    alpha=0.15, color="purple")
    ax.set_title("Delhi - Actual vs Predicted AQI (2025)\nCNN-LSTM Model",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "timeseries_delhi.png"), dpi=150)
    plt.show()
    print("  Saved -> outputs/timeseries_delhi.png")

# ── Save final model ──────────────────────────────────────────────────────────
model.save(os.path.join(MODEL_DIR, "cnn_lstm_final.h5"))
print("  Model saved -> models/cnn_lstm_final.h5")

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"  RMSE : {rmse:.2f}")
print(f"  MAE  : {mae:.2f}")
print(f"  R2   : {r2:.4f}")
print(f"  Train: {len(X_train)} samples")
print(f"  Test : {len(X_test)} samples")
print("="*50)
print("\n=== Training Complete ===")