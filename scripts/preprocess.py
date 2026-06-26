import pandas as pd
import numpy as np
import os
import rasterio
from scipy.interpolate import RegularGridInterpolator

# ── Paths ─────────────────────────────────────────────────────────────────────
CPCB_DIR    = r"C:\Users\Victus\OneDrive\Desktop\AQI\data\cpcb"
TROPOMI_DIR = r"C:\Users\Victus\OneDrive\Desktop\AQI\data\tropomi"
OUTPUT_DIR  = r"C:\Users\Victus\OneDrive\Desktop\AQI\data\processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Station coordinates ───────────────────────────────────────────────────────
STATION_INFO = {
    "Narela_Delhi":  {"lat": 28.8560, "lon": 77.1010, "state": "Delhi"},
    "Jaipur":        {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Bengaluru":     {"lat": 12.9121, "lon": 77.6097, "state": "Karnataka"},
    "Amritsar":      {"lat": 31.6340, "lon": 74.8723, "state": "Punjab"},
    "Lucknow":       {"lat": 26.8467, "lon": 80.9462, "state": "UP"},
    "Mumbai":        {"lat": 19.2307, "lon": 72.8567, "state": "Maharashtra"},
    "Kolkata":       {"lat": 22.5154, "lon": 88.3632, "state": "WestBengal"},
    "Gurugram":      {"lat": 28.4595, "lon": 77.0266, "state": "Haryana"},
    "Chandigarh":    {"lat": 30.7333, "lon": 76.7794, "state": "Punjab"},
}

FILE_TO_STATION = {
    "Raw_data_1Day_2025_site_1426_Narela_Delhi_DPCC_1Day.csv":                               "Narela_Delhi",
    "Raw_data_1Day_2025_site_134_Police_Commissionerate_Jaipur_RSPCB_1Day.csv":              "Jaipur",
    "Raw_data_1Day_2025_site_162_BTM_Layout_Bengaluru_CPCB_1Day.csv":                        "Bengaluru",
    "Raw_data_1Day_2025_site_256_Golden_Temple_Amritsar_PPCB_1Day.csv":                      "Amritsar",
    "Raw_data_1Day_2025_site_297_Talkatora_District_Industries_Center_Lucknow_CPCB_1Day.csv":"Lucknow",
    "Raw_data_1Day_2025_site_5113_Borivali_East_Mumbai_MPCB_1Day.csv":                       "Mumbai",
    "Raw_data_1Day_2025_site_5126_Rabindra_Sarobar_Kolkata_WBPCB_1Day.csv":                  "Kolkata",
    "Raw_data_1Day_2025_site_5344_Teri_Gram_Gurugram_HSPCB_1Day.csv":                        "Gurugram",
    "Raw_data_1Day_2025_site_5582_Sector-53_Chandigarh_CPCC_1Day.csv":                       "Chandigarh",
}

# ── Step 1: Load CPCB data ────────────────────────────────────────────────────
print("=== Step 1: Loading CPCB station data ===")
all_stations = []

for filename, station_key in FILE_TO_STATION.items():
    filepath = os.path.join(CPCB_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  WARNING: File not found → {filename}")
        continue

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Timestamp":      "date",
        "PM2.5 (µg/m³)": "PM25",
        "NO2 (µg/m³)":   "NO2",
        "CO (mg/m³)":     "CO",
        "Ozone (µg/m³)":  "O3",
        "SO2 (µg/m³)":   "SO2",
        "AT (°C)":        "temp",
        "RH (%)":         "rh",
        "WS (m/s)":       "wind_speed",
        "WD (deg)":       "wind_dir",
    })

    keep_cols = ["date", "PM25", "NO2", "CO", "O3", "SO2",
                 "temp", "rh", "wind_speed", "wind_dir"]
    df = df[[c for c in keep_cols if c in df.columns]]

    info = STATION_INFO[station_key]
    df["station"] = station_key
    df["lat"]     = info["lat"]
    df["lon"]     = info["lon"]
    df["state"]   = info["state"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.replace("NA", np.nan)

    for col in ["PM25", "NO2", "CO", "O3", "SO2",
                "temp", "rh", "wind_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    all_stations.append(df)
    print(f"  Loaded {station_key} → {len(df)} rows")

cpcb_all = pd.concat(all_stations, ignore_index=True)
print(f"\nTotal rows: {len(cpcb_all)}")
print(f"Date range: {cpcb_all['date'].min()} to {cpcb_all['date'].max()}")

# ── Step 2: Fix missing values ────────────────────────────────────────────────
print("\n=== Step 2: Fixing missing values ===")

cpcb_all["month"] = cpcb_all["date"].dt.month

# Fill temp by monthly mean per station, then overall station mean
for col in ["temp", "rh", "wind_speed", "wind_dir", "NO2", "CO", "O3", "SO2"]:
    if col in cpcb_all.columns:
        # Monthly mean per station
        cpcb_all[col] = cpcb_all.groupby(
            ["station", "month"]
        )[col].transform(lambda x: x.fillna(x.mean()))
        # Overall station mean for remaining
        cpcb_all[col] = cpcb_all.groupby(
            "station"
        )[col].transform(lambda x: x.fillna(x.mean()))

print(f"  Missing temp after fix:  {cpcb_all['temp'].isna().sum()}")
print(f"  Missing NO2 after fix:   {cpcb_all['NO2'].isna().sum()}")
print(f"  Missing PM25:            {cpcb_all['PM25'].isna().sum()}")

# ── Step 3: Compute AQI ───────────────────────────────────────────────────────
print("\n=== Step 3: Computing AQI from PM2.5 ===")

def pm25_to_aqi(pm25):
    if pd.isna(pm25):
        return np.nan
    breakpoints = [
        (0,   30,   0,   50),
        (30,  60,   51,  100),
        (60,  90,   101, 200),
        (90,  120,  201, 300),
        (120, 250,  301, 400),
        (250, 500,  401, 500),
    ]
    for (c_lo, c_hi, i_lo, i_hi) in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo
    return 500

cpcb_all["AQI"] = cpcb_all["PM25"].apply(pm25_to_aqi)
print(f"  AQI mean: {cpcb_all['AQI'].mean():.1f}")
print(f"  AQI max:  {cpcb_all['AQI'].max():.1f}")

# ── Step 4: Load satellite data per month ─────────────────────────────────────
print("\n=== Step 4: Loading satellite data per month ===")

def load_tif_interpolator(tif_path):
    """Load a GeoTIFF and return a scipy interpolator for lat/lon lookup"""
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        bounds = src.bounds
        nrows, ncols = data.shape
        lats = np.linspace(bounds.top, bounds.bottom, nrows)
        lons = np.linspace(bounds.left, bounds.right, ncols)
    interp = RegularGridInterpolator(
        (lats[::-1], lons), data[::-1],
        method="linear", bounds_error=False, fill_value=np.nan
    )
    return interp

# Find all available satellite months
sat_months = {}
for pollutant in ["NO2", "SO2", "CO", "O3", "HCHO"]:
    folder = os.path.join(TROPOMI_DIR, pollutant)
    if not os.path.exists(folder):
        continue
    for f in os.listdir(folder):
        if f.endswith("_mean.tif"):
            # Extract YYYYMM from filename e.g. no2_202501_mean.tif
            parts = f.split("_")
            if len(parts) >= 2:
                yyyymm = parts[1]
                if yyyymm not in sat_months:
                    sat_months[yyyymm] = {}
                sat_months[yyyymm][pollutant] = os.path.join(folder, f)

print(f"  Found satellite months: {sorted(sat_months.keys())}")

# Build interpolators for each month
sat_interps = {}
for yyyymm, files in sat_months.items():
    sat_interps[yyyymm] = {}
    for pollutant, path in files.items():
        try:
            sat_interps[yyyymm][pollutant] = load_tif_interpolator(path)
            print(f"  Loaded {pollutant} {yyyymm}")
        except Exception as e:
            print(f"  ERROR loading {pollutant} {yyyymm}: {e}")

# ── Step 5: Extract satellite values per row ──────────────────────────────────
print("\n=== Step 5: Extracting satellite values per row ===")

cpcb_all["yyyymm"] = cpcb_all["date"].dt.strftime("%Y%m")

def extract_sat_value(row, pollutant):
    yyyymm = row["yyyymm"]
    # Try exact month match first
    if yyyymm in sat_interps and pollutant in sat_interps[yyyymm]:
        val = float(sat_interps[yyyymm][pollutant](
            [[row["lat"], row["lon"]]]
        )[0])
        return val
    # Fall back to nearest available month
    available = sorted(sat_interps.keys())
    if available:
        nearest = min(available, key=lambda x: abs(int(x) - int(yyyymm)))
        if pollutant in sat_interps[nearest]:
            val = float(sat_interps[nearest][pollutant](
                [[row["lat"], row["lon"]]]
            )[0])
            return val
    return np.nan

for pollutant, col_name in [
    ("NO2",  "NO2_sat"),
    ("SO2",  "SO2_sat"),
    ("CO",   "CO_sat"),
    ("O3",   "O3_sat"),
    ("HCHO", "HCHO_sat"),
]:
    print(f"  Extracting {col_name}...")
    cpcb_all[col_name] = cpcb_all.apply(
        lambda row: extract_sat_value(row, pollutant), axis=1
    )
    print(f"    Mean: {cpcb_all[col_name].mean():.4e}, "
          f"Missing: {cpcb_all[col_name].isna().sum()}")

# ── Step 6: Temporal features ─────────────────────────────────────────────────
print("\n=== Step 6: Adding temporal features ===")

cpcb_all["dayofyear"] = cpcb_all["date"].dt.dayofyear
cpcb_all["dayofweek"] = cpcb_all["date"].dt.dayofweek

cpcb_all["month_sin"] = np.sin(2 * np.pi * cpcb_all["month"] / 12)
cpcb_all["month_cos"] = np.cos(2 * np.pi * cpcb_all["month"] / 12)
cpcb_all["doy_sin"]   = np.sin(2 * np.pi * cpcb_all["dayofyear"] / 365)
cpcb_all["doy_cos"]   = np.cos(2 * np.pi * cpcb_all["dayofyear"] / 365)

# Season flag
def get_season(month):
    if month in [12, 1, 2]:  return 0  # Winter
    elif month in [3, 4, 5]: return 1  # Summer
    elif month in [6, 7, 8, 9]: return 2  # Monsoon
    else: return 3  # Post-Monsoon

cpcb_all["season"] = cpcb_all["month"].apply(get_season)

# Lag features
cpcb_all = cpcb_all.sort_values(["station", "date"])
cpcb_all["PM25_lag1"]  = cpcb_all.groupby("station")["PM25"].shift(1)
cpcb_all["PM25_lag3"]  = cpcb_all.groupby("station")["PM25"].shift(3)
cpcb_all["PM25_lag7"]  = cpcb_all.groupby("station")["PM25"].shift(7)
cpcb_all["PM25_roll3"] = cpcb_all.groupby("station")["PM25"].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
cpcb_all["PM25_roll7"] = cpcb_all.groupby("station")["PM25"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
cpcb_all["AQI_lag1"]   = cpcb_all.groupby("station")["AQI"].shift(1)

print(f"  Total features: {len(cpcb_all.columns)}")

# ── Step 7: Save ──────────────────────────────────────────────────────────────
print("\n=== Step 7: Saving ===")

final_df = cpcb_all.dropna(subset=["AQI", "PM25"])
print(f"  Rows after dropping missing AQI: {len(final_df)}")

output_path = os.path.join(OUTPUT_DIR, "processed_dataset.csv")
final_df.to_csv(output_path, index=False)
print(f"  Saved → {output_path}")

print("\n=== Dataset Summary ===")
print(f"Total rows:   {len(final_df)}")
print(f"Stations:     {final_df['station'].nunique()}")
print(f"Date range:   {final_df['date'].min()} to {final_df['date'].max()}")
print(f"AQI mean:     {final_df['AQI'].mean():.1f}")
print(f"AQI max:      {final_df['AQI'].max():.1f}")
print(f"Missing temp: {final_df['temp'].isna().sum()}")
print("\n=== Preprocessing Complete ===")