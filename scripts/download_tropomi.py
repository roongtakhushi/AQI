import ee
import geemap
import os

ee.Initialize(project='aqi-project-500016')

INDIA_BBOX = ee.Geometry.Rectangle([68.0, 8.0, 97.5, 37.5])
OUTPUT_DIR = r"C:\Users\Victus\OneDrive\Desktop\AQI\data\tropomi"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = {
    "NO2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_NO2",
        "band": "tropospheric_NO2_column_number_density",
        "prefix": "no2",
        "scale": 10000
    },
    "SO2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_SO2",
        "band": "SO2_column_number_density",
        "prefix": "so2",
        "scale": 10000
    },
    "CO": {
        "collection": "COPERNICUS/S5P/OFFL/L3_CO",
        "band": "CO_column_number_density",
        "prefix": "co",
        "scale": 10000
    },
    "O3": {
        "collection": "COPERNICUS/S5P/OFFL/L3_O3",
        "band": "O3_column_number_density",
        "prefix": "o3",
        "scale": 10000
    },
    "HCHO": {
        "collection": "COPERNICUS/S5P/OFFL/L3_HCHO",
        "band": "tropospheric_HCHO_column_number_density",
        "prefix": "hcho",
        "scale": 10000
    },
}

def download_monthly_mean(pollutant, config, year, month):
    import calendar
    start = f"{year}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    end = f"{year}-{month:02d}-{last_day}"

    print(f"  Downloading {pollutant} — {year}-{month:02d} ...", end=" ")

    collection = ee.ImageCollection(config["collection"]) \
        .filterDate(start, end) \
        .filterBounds(INDIA_BBOX)

    if collection.size().getInfo() == 0:
        print("No data found, skipping.")
        return

    mean_image = collection.select(config["band"]).mean().clip(INDIA_BBOX)

    out_folder = os.path.join(OUTPUT_DIR, pollutant)
    os.makedirs(out_folder, exist_ok=True)

    filename = os.path.join(out_folder, f"{config['prefix']}_{year}{month:02d}_mean.tif")

    geemap.ee_export_image(
        mean_image,
        filename=filename,
        scale=config["scale"],
        region=INDIA_BBOX,
        crs="EPSG:4326"
    )
    print(f"saved → {filename}")

print("=== Starting TROPOMI download ===")

# ── CHANGED: 2025, months 1-3 to match your CPCB data ──
for pollutant, config in DATASETS.items():
    print(f"\n[{pollutant}]")
    for month in [1, 2, 3]:          # Jan, Feb, Mar 2025
        try:
            download_monthly_mean(pollutant, config, 2025, month)
        except Exception as e:
            print(f"  ERROR: {e}")

print("\n=== Download complete ===")