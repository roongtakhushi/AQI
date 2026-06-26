import numpy as np
import pandas as pd
import os
import joblib
import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import tensorflow as tf
from scipy.interpolate import RegularGridInterpolator

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = r"C:\Users\Victus\OneDrive\Desktop\AQI\models"
TROPOMI_DIR = r"C:\Users\Victus\OneDrive\Desktop\AQI\data\tropomi"
OUTPUT_DIR  = r"C:\Users\Victus\OneDrive\Desktop\AQI\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model and scalers ────────────────────────────────────────────────────
print("=== Loading model and scalers ===")
model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "cnn_lstm_best.h5"),
    compile=False
)
model.compile(optimizer="adam", loss="huber", metrics=["mae"])
scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))
print("  Model and scalers loaded")

# ── Features list — must match train_model.py exactly ────────────────────────
FEATURES = [
    "NO2_sat", "SO2_sat", "CO_sat", "O3_sat", "HCHO_sat",
    "NO2", "CO", "O3", "SO2", "temp", "rh", "wind_speed",
    "lat", "lon",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "dayofweek", "season",
    "PM25_lag1", "PM25_lag3", "PM25_lag7",
    "PM25_roll3", "PM25_roll7", "AQI_lag1",
]

# ── Create India grid ─────────────────────────────────────────────────────────
print("\n=== Creating India grid ===")
lat_grid = np.arange(8.0,  37.5, 0.25)
lon_grid = np.arange(68.0, 97.5, 0.25)
lons, lats = np.meshgrid(lon_grid, lat_grid)
n_points = lats.size
print(f"  Grid shape: {lats.shape} ({n_points} points)")

# ── Load satellite data ───────────────────────────────────────────────────────
print("\n=== Loading satellite data ===")

def load_tif_interpolator(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(float)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        bounds = src.bounds
        nrows, ncols = data.shape
        src_lats = np.linspace(bounds.top,  bounds.bottom, nrows)
        src_lons = np.linspace(bounds.left, bounds.right,  ncols)
    interp = RegularGridInterpolator(
        (src_lats[::-1], src_lons), data[::-1],
        method="linear", bounds_error=False, fill_value=np.nan
    )
    return interp

def get_best_sat_file(pollutant, preferred_yyyymm="202501"):
    """Find satellite file — prefer Jan 2025, fallback to any available"""
    folder = os.path.join(TROPOMI_DIR, pollutant)
    if not os.path.exists(folder):
        return None
    # Try preferred month first
    for f in os.listdir(folder):
        if preferred_yyyymm in f and f.endswith(".tif"):
            return os.path.join(folder, f)
    # Fallback to any available file
    files = [f for f in os.listdir(folder) if f.endswith(".tif")]
    if files:
        return os.path.join(folder, sorted(files)[0])
    return None

# Load interpolators for each pollutant
sat_interps = {}
for pollutant in ["NO2", "SO2", "CO", "O3", "HCHO"]:
    path = get_best_sat_file(pollutant)
    if path:
        sat_interps[pollutant] = load_tif_interpolator(path)
        print(f"  Loaded {pollutant} from {os.path.basename(path)}")
    else:
        sat_interps[pollutant] = None
        print(f"  WARNING: No file found for {pollutant}")

# Extract satellite values for every grid point
grid_points = np.column_stack([lats.ravel(), lons.ravel()])
sat_values  = {}
for pollutant, interp in sat_interps.items():
    if interp is not None:
        vals = interp(grid_points)
        # Fill NaN with median
        median_val = np.nanmedian(vals)
        vals = np.where(np.isnan(vals), median_val, vals)
        sat_values[pollutant] = vals
    else:
        sat_values[pollutant] = np.zeros(n_points)

print(f"  NO2 range:  {sat_values['NO2'].min():.4e} to {sat_values['NO2'].max():.4e}")
print(f"  HCHO range: {sat_values['HCHO'].min():.4e} to {sat_values['HCHO'].max():.4e}")

# ── Load processed data for realistic feature values ─────────────────────────
print("\n=== Loading reference data for feature means ===")
DATA_PATH = r"C:\Users\Victus\OneDrive\Desktop\AQI\data\processed\processed_dataset.csv"
ref_df = pd.read_csv(DATA_PATH)

# January mean values for meteorological features
jan_df  = ref_df[ref_df["month"] == 1] if "month" in ref_df.columns else ref_df
temp_mean       = jan_df["temp"].mean()       if "temp" in jan_df.columns else 15.0
rh_mean         = jan_df["rh"].mean()         if "rh" in jan_df.columns else 65.0
wind_mean       = jan_df["wind_speed"].mean() if "wind_speed" in jan_df.columns else 2.0
no2_mean        = jan_df["NO2"].mean()        if "NO2" in jan_df.columns else 30.0
co_mean         = jan_df["CO"].mean()         if "CO" in jan_df.columns else 1.0
o3_mean         = jan_df["O3"].mean()         if "O3" in jan_df.columns else 50.0
so2_mean        = jan_df["SO2"].mean()        if "SO2" in jan_df.columns else 10.0
pm25_mean       = jan_df["PM25"].mean()       if "PM25" in jan_df.columns else 80.0
aqi_mean        = jan_df["AQI"].mean()        if "AQI" in jan_df.columns else 150.0

print(f"  Reference temp:  {temp_mean:.1f}°C")
print(f"  Reference RH:    {rh_mean:.1f}%")
print(f"  Reference PM2.5: {pm25_mean:.1f} µg/m³")
print(f"  Reference AQI:   {aqi_mean:.1f}")

# ── Build feature grid ────────────────────────────────────────────────────────
print("\n=== Building feature grid ===")

month     = 1
doy       = 15
dayofweek = 0
season    = 0  # Winter

feature_grid = np.column_stack([
    # Satellite
    sat_values["NO2"],
    sat_values["SO2"],
    sat_values["CO"],
    sat_values["O3"],
    sat_values["HCHO"],
    # Ground met — use realistic January climatology
    np.full(n_points, no2_mean),
    np.full(n_points, co_mean),
    np.full(n_points, o3_mean),
    np.full(n_points, so2_mean),
    np.full(n_points, temp_mean),
    np.full(n_points, rh_mean),
    np.full(n_points, wind_mean),
    # Location
    lats.ravel(),
    lons.ravel(),
    # Temporal
    np.full(n_points, np.sin(2 * np.pi * month / 12)),
    np.full(n_points, np.cos(2 * np.pi * month / 12)),
    np.full(n_points, np.sin(2 * np.pi * doy / 365)),
    np.full(n_points, np.cos(2 * np.pi * doy / 365)),
    np.full(n_points, dayofweek),
    np.full(n_points, season),
    # Lag features — use realistic mean
    np.full(n_points, pm25_mean),
    np.full(n_points, pm25_mean),
    np.full(n_points, pm25_mean),
    np.full(n_points, pm25_mean),
    np.full(n_points, pm25_mean),
    np.full(n_points, aqi_mean),
])

print(f"  Feature grid shape: {feature_grid.shape}")
print(f"  Features: {feature_grid.shape[1]} (should be {len(FEATURES)})")

# ── Predict AQI ───────────────────────────────────────────────────────────────
print("\n=== Predicting AQI across India ===")

X_scaled = scaler_X.transform(feature_grid)
X_lstm   = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

y_pred_scaled = model.predict(X_lstm, batch_size=512, verbose=1).flatten()
y_pred_aqi    = scaler_y.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).flatten()
y_pred_aqi    = np.clip(y_pred_aqi, 0, 500)

aqi_map = y_pred_aqi.reshape(lats.shape)

print(f"  AQI min:  {aqi_map.min():.1f}")
print(f"  AQI max:  {aqi_map.max():.1f}")
print(f"  AQI mean: {aqi_map.mean():.1f}")

# Save AQI grid as CSV
aqi_df = pd.DataFrame({
    "lat": lats.ravel(),
    "lon": lons.ravel(),
    "AQI": y_pred_aqi
})
aqi_df.to_csv(os.path.join(OUTPUT_DIR, "aqi_grid.csv"), index=False)
print("  AQI grid saved → outputs/aqi_grid.csv")

# ── Plot AQI map ──────────────────────────────────────────────────────────────
print("\n=== Generating AQI map ===")

def make_aqi_map(aqi_grid, lats, lons, title, filename):
    fig, ax = plt.subplots(figsize=(12, 13),
                            subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([68, 97, 8, 37], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND,      facecolor="#f5f5f5")
    ax.add_feature(cfeature.OCEAN,     facecolor="#c8e6f5")
    ax.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS,   linewidth=1.0, edgecolor="black")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")

    im = ax.contourf(
        lons, lats, aqi_grid,
        levels=[0, 50, 100, 150, 200, 300, 400, 500],
        colors=["#00e400", "#92d050", "#ffff00",
                "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"],
        transform=ccrs.PlateCarree(),
        alpha=0.85,
        extend="max"
    )

    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                         pad=0.02, shrink=0.75, aspect=20)
    cbar.set_label("Air Quality Index (AQI)", fontsize=11)
    cbar.set_ticks([0, 50, 100, 150, 200, 300, 400, 500])
    cbar.set_ticklabels([
        "0", "50\nGood", "100\nSatisfactory",
        "150\nModerate", "200\nPoor",
        "300\nVery Poor", "400\nSevere", "500"
    ])

    # City markers
    cities = {
        "Delhi":     (28.61, 77.20),
        "Lucknow":   (26.85, 80.95),
        "Kolkata":   (22.57, 88.36),
        "Mumbai":    (19.08, 72.88),
        "Bengaluru": (12.97, 77.59),
        "Amritsar":  (31.63, 74.87),
        "Chandigarh":(30.73, 76.78),
        "Jaipur":    (26.91, 75.79),
        "Gurugram":  (28.46, 77.03),
    }
    for city, (lat, lon) in cities.items():
        ax.plot(lon, lat, "k^", markersize=6,
                transform=ccrs.PlateCarree(), zorder=5)
        ax.text(lon + 0.3, lat + 0.1, city, fontsize=7.5,
                transform=ccrs.PlateCarree(),
                fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.1",
                          facecolor="white", alpha=0.6, edgecolor="none"))

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # AQI stats box
    stats_text = (f"Mean AQI: {aqi_grid.mean():.0f}\n"
                  f"Max AQI:  {aqi_grid.max():.0f}\n"
                  f"Min AQI:  {aqi_grid.min():.0f}")
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white",
                      alpha=0.8, edgecolor="gray"))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {out_path}")

# January 2025 map
make_aqi_map(
    aqi_map, lats, lons,
    title="Predicted Surface AQI over India — January 2025\n(CNN-LSTM | TROPOMI + CPCB)",
    filename="aqi_map_jan2025.png"
)

# ── Generate maps for all 3 months ───────────────────────────────────────────
print("\n=== Generating maps for all available months ===")

month_configs = [
    (1,  15,  0, "January 2025",   "Winter",      "aqi_map_jan2025_v2.png"),
    (2,  46,  0, "February 2025",  "Winter",      "aqi_map_feb2025.png"),
    (3,  74,  0, "March 2025",     "Post-Monsoon","aqi_map_mar2025.png"),
]

for (month, doy, season_id, month_name, season_name, fname) in month_configs:

    # Get satellite file for this month
    yyyymm = f"2025{month:02d}"
    for pollutant in ["NO2", "SO2", "CO", "O3", "HCHO"]:
        path = get_best_sat_file(pollutant, preferred_yyyymm=yyyymm)
        if path:
            sat_interps[pollutant] = load_tif_interpolator(path)

    # Rebuild satellite values
    for pollutant, interp in sat_interps.items():
        if interp is not None:
            vals = interp(grid_points)
            median_val = np.nanmedian(vals)
            vals = np.where(np.isnan(vals), median_val, vals)
            sat_values[pollutant] = vals

    # Get month-specific met means from training data
    m_df     = ref_df[ref_df["month"] == month] if "month" in ref_df.columns else ref_df
    t_mean   = m_df["temp"].mean()        if "temp" in m_df.columns else temp_mean
    r_mean   = m_df["rh"].mean()          if "rh" in m_df.columns else rh_mean
    w_mean   = m_df["wind_speed"].mean()  if "wind_speed" in m_df.columns else wind_mean
    pm_mean  = m_df["PM25"].mean()        if "PM25" in m_df.columns else pm25_mean
    aq_mean  = m_df["AQI"].mean()         if "AQI" in m_df.columns else aqi_mean
    n2_mean  = m_df["NO2"].mean()         if "NO2" in m_df.columns else no2_mean
    c_mean   = m_df["CO"].mean()          if "CO" in m_df.columns else co_mean
    o_mean   = m_df["O3"].mean()          if "O3" in m_df.columns else o3_mean
    s_mean   = m_df["SO2"].mean()         if "SO2" in m_df.columns else so2_mean

    fg = np.column_stack([
        sat_values["NO2"], sat_values["SO2"],
        sat_values["CO"],  sat_values["O3"], sat_values["HCHO"],
        np.full(n_points, n2_mean), np.full(n_points, c_mean),
        np.full(n_points, o_mean),  np.full(n_points, s_mean),
        np.full(n_points, t_mean),  np.full(n_points, r_mean),
        np.full(n_points, w_mean),
        lats.ravel(), lons.ravel(),
        np.full(n_points, np.sin(2 * np.pi * month / 12)),
        np.full(n_points, np.cos(2 * np.pi * month / 12)),
        np.full(n_points, np.sin(2 * np.pi * doy / 365)),
        np.full(n_points, np.cos(2 * np.pi * doy / 365)),
        np.full(n_points, 0),
        np.full(n_points, season_id),
        np.full(n_points, pm_mean), np.full(n_points, pm_mean),
        np.full(n_points, pm_mean), np.full(n_points, pm_mean),
        np.full(n_points, pm_mean), np.full(n_points, aq_mean),
    ])

    Xs = scaler_X.transform(fg)
    Xl = Xs.reshape(Xs.shape[0], 1, Xs.shape[1])
    yp = scaler_y.inverse_transform(
        model.predict(Xl, batch_size=512, verbose=0).reshape(-1, 1)
    ).flatten()
    yp = np.clip(yp, 0, 500).reshape(lats.shape)

    make_aqi_map(
        yp, lats, lons,
        title=f"Predicted Surface AQI over India — {month_name}\n"
              f"(CNN-LSTM | TROPOMI + CPCB | {season_name})",
        filename=fname
    )

print("\n=== predict_aqi.py Complete ===")
print("Files saved in outputs/:")
print("  aqi_map_jan2025.png")
print("  aqi_map_feb2025.png")
print("  aqi_map_mar2025.png")
print("  aqi_grid.csv")