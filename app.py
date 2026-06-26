import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# Set page configuration with rich title and layout
st.set_page_config(
    page_title="AQI India Dashboard & Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design aesthetics (glassmorphism, premium colors, custom fonts)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(135deg, #1fa2ff, #12d6df, #a6ffcb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-weight: 400;
        font-size: 1.2rem;
        color: #a0aec0;
        margin-bottom: 2rem;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }
    
    /* AQI Categories styling */
    .aqi-good { color: #00e400; }
    .aqi-satisfactory { color: #92d050; }
    .aqi-moderate { color: #ffff00; }
    .aqi-poor { color: #ff7e00; }
    .aqi-very-poor { color: #ff0000; }
    .aqi-severe { color: #8f3f97; }
</style>
""", unsafe_allow_html=True)

# ── Paths (relative for deployment portability) ──────────────────────────────
MODEL_DIR = "models"
DATA_PATH = os.path.join("data", "processed", "processed_dataset.csv")
OUTPUT_DIR = "outputs"

# ── Load model and scalers (Cached to only run once) ──────────────────────────
@st.cache_resource
def load_ml_assets():
    try:
        model_path = os.path.join(MODEL_DIR, "cnn_lstm_best.h5")
        scaler_X_path = os.path.join(MODEL_DIR, "scaler_X.pkl")
        scaler_y_path = os.path.join(MODEL_DIR, "scaler_y.pkl")
        
        # Load keras model
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss="huber", metrics=["mae"])
        
        # Load scalers
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        return model, scaler_X, scaler_y, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler_X, scaler_y, error_msg = load_ml_assets()

# ── Load Reference Dataset ────────────────────────────────────────────────────
@st.cache_data
def load_reference_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    return None

ref_df = load_reference_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Air Quality Index (AQI) Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Spatial Mapping and Real-time Deep Learning Predictions over India using CNN-LSTM & Satellite Data</div>', unsafe_allow_html=True)

if error_msg:
    st.error(f"Error loading machine learning assets from '{MODEL_DIR}': {error_msg}")
    st.warning("Please make sure the models folder containing 'cnn_lstm_best.h5', 'scaler_X.pkl', and 'scaler_y.pkl' is uploaded correctly to your repository.")

# ── Tabs Configuration ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Spatial AQI Maps (India)", 
    "🔮 Real-Time AQI Calculator", 
    "📊 Model Performance & Insights",
    "🔍 Custom Cross-Verification"
])

# ── TAB 1: Spatial AQI Maps (India) ───────────────────────────────────────────
with tab1:
    st.header("Spatial AQI Mapping over India (Jan - Mar 2025)")
    st.write(
        "These maps visualize predicted surface AQI across India, generated using our CNN-LSTM deep learning model combining TROPOMI satellite columns and ground-based meteorological/station data."
    )
    
    month_selection = st.radio(
        "Select Prediction Month:",
        ["January 2025", "February 2025", "March 2025"],
        horizontal=True
    )
    
    # Map map names to pre-generated files
    map_files = {
        "January 2025": "aqi_map_jan2025.png",
        "February 2025": "aqi_map_feb2025.png",
        "March 2025": "aqi_map_mar2025.png"
    }
    
    filename = map_files[month_selection]
    map_path = os.path.join(OUTPUT_DIR, filename)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if os.path.exists(map_path):
            st.image(map_path, use_container_width=True, caption=f"AQI Spatial Distribution for {month_selection}")
        else:
            st.warning(f"Map file '{filename}' not found in the 'outputs/' folder. Showing placeholder or please ensure outputs are uploaded.")
            st.info("Ensure files exist in: `outputs/` directory.")
            
    with col2:
        st.subheader("Key Findings & Legend")
        
        # AQI levels table/explanation
        st.markdown("""
        | AQI Range | Class | Health Impact |
        | :--- | :--- | :--- |
        | <span class="aqi-good">**0 - 50**</span> | Good | Minimal impact |
        | <span class="aqi-satisfactory">**51 - 100**</span> | Satisfactory | Minor breathing discomfort to sensitive people |
        | <span class="aqi-moderate">**101 - 200**</span> | Moderate | Breathing discomfort to people with lungs, asthma and heart diseases |
        | <span class="aqi-poor">**201 - 300**</span> | Poor | Breathing discomfort to most people on prolonged exposure |
        | <span class="aqi-very-poor">**301 - 400**</span> | Very Poor | Respiratory illness on prolonged exposure |
        | <span class="aqi-severe">**401 - 500**</span> | Severe | Affects healthy people and seriously impacts those with existing diseases |
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### Spatial Insights
        * **Northern India & Indo-Gangetic Plains (IGP)**: Typically experience higher AQI levels (frequently scaling into the Moderate, Poor, or Severe categories) due to meteorological factors like low temperatures, low wind speeds, boundary layer compression, and satellite observations indicating dense $NO_2$ and particulate loadings.
        * **Peninsular & Coastal India**: Experience better air quality (typically Good or Satisfactory) facilitated by sea breezes and favorable dispersion conditions.
        """)

# ── TAB 2: Real-Time AQI Calculator ───────────────────────────────────────────
with tab2:
    st.header("Real-Time AQI Deep Learning Calculator")
    st.write(
        "Select an monitoring station or specify meteorological and satellite values below to calculate the predicted AQI in real-time using our trained Keras CNN-LSTM model."
    )
    
    if ref_df is not None:
        # User can pick a station to load defaults
        unique_stations = sorted(ref_df["station"].unique())
        selected_station = st.selectbox(
            "Quick Select Monitoring Station (loads defaults):",
            unique_stations,
            index=unique_stations.index("Delhi") if "Delhi" in unique_stations else 0
        )
        
        # Get defaults from selected station
        station_df = ref_df[ref_df["station"] == selected_station]
        # Get mean values for the station
        defaults = station_df.mean(numeric_only=True).to_dict()
        defaults["lat"] = station_df["lat"].iloc[0] if "lat" in station_df.columns else 28.6
        defaults["lon"] = station_df["lon"].iloc[0] if "lon" in station_df.columns else 77.2
    else:
        st.info("Reference dataset not found at `data/processed/processed_dataset.csv`. Using default regional values.")
        defaults = {
            "lat": 28.61, "lon": 77.20,
            "temp": 15.0, "rh": 65.0, "wind_speed": 1.5,
            "NO2_sat": 5e-5, "SO2_sat": 2e-4, "CO_sat": 4e-2, "O3_sat": 1.2e-1, "HCHO_sat": 1.5e-4,
            "NO2": 30.0, "CO": 1.0, "O3": 50.0, "SO2": 10.0,
            "PM25_lag1": 80.0, "AQI_lag1": 150.0
        }

    # Setup columns for parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌐 Location & Time")
        lat = st.number_input("Latitude (°N)", min_value=8.0, max_value=38.0, value=float(defaults.get("lat", 28.61)), step=0.01)
        lon = st.number_input("Longitude (°E)", min_value=68.0, max_value=98.0, value=float(defaults.get("lon", 77.20)), step=0.01)
        
        month = st.slider("Month of Year", 1, 12, 1)
        day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
        season = st.selectbox("Season", ["Winter", "Summer", "Monsoon", "Post-Monsoon"])
        season_mapping = {"Winter": 0, "Summer": 1, "Monsoon": 2, "Post-Monsoon": 3}
        season_id = season_mapping[season]

    with col2:
        st.subheader("🌦️ Meteorological Parameters")
        temp = st.slider("Temperature (°C)", -5.0, 50.0, float(defaults.get("temp", 15.0)), step=0.1)
        rh = st.slider("Relative Humidity (%)", 0.0, 100.0, float(defaults.get("rh", 65.0)), step=0.5)
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 15.0, float(defaults.get("wind_speed", 2.0)), step=0.1)
        
        st.subheader("📡 TROPOMI Satellite Columns")
        no2_sat = st.number_input("NO2 Column Amount", value=float(defaults.get("NO2_sat", 5e-5)), format="%.3e")
        so2_sat = st.number_input("SO2 Column Amount", value=float(defaults.get("SO2_sat", 2e-4)), format="%.3e")
        co_sat = st.number_input("CO Column Amount", value=float(defaults.get("CO_sat", 4e-2)), format="%.3e")
        o3_sat = st.number_input("O3 Column Amount", value=float(defaults.get("O3_sat", 1.2e-1)), format="%.3e")
        hcho_sat = st.number_input("HCHO Column Amount", value=float(defaults.get("HCHO_sat", 1.5e-4)), format="%.3e")

    with col3:
        st.subheader("🏫 Ground Station Context")
        no2_gr = st.slider("Ground NO₂ (µg/m³)", 0.0, 250.0, float(defaults.get("NO2", 30.0)))
        co_gr = st.slider("Ground CO (mg/m³)", 0.0, 10.0, float(defaults.get("CO", 1.0)), step=0.1)
        o3_gr = st.slider("Ground O₃ (µg/m³)", 0.0, 200.0, float(defaults.get("O3", 50.0)))
        so2_gr = st.slider("Ground SO₂ (µg/m³)", 0.0, 150.0, float(defaults.get("SO2", 10.0)))
        
        st.subheader("🕒 Historical (Lag) Values")
        pm25_lag = st.slider("Yesterday PM2.5 (µg/m³)", 0.0, 500.0, float(defaults.get("PM25_lag1", 80.0)))
        aqi_lag = st.slider("Yesterday AQI", 0.0, 500.0, float(defaults.get("AQI_lag1", 150.0)))

    # Run Prediction
    st.markdown("---")
    
    if st.button("🔮 Predict Air Quality Index", use_container_width=True):
        if model is None:
            st.error("Cannot compute prediction: ML Model assets are not loaded.")
        else:
            with st.spinner("Calculating AQI..."):
                # Reconstruct features in the exact expected order:
                # ── FEATURES list (26 features) ──
                # "NO2_sat", "SO2_sat", "CO_sat", "O3_sat", "HCHO_sat",
                # "NO2", "CO", "O3", "SO2", "temp", "rh", "wind_speed",
                # "lat", "lon",
                # "month_sin", "month_cos", "doy_sin", "doy_cos",
                # "dayofweek", "season",
                # "PM25_lag1", "PM25_lag3", "PM25_lag7",
                # "PM25_roll3", "PM25_roll7", "AQI_lag1"
                
                # Approximate day of year from month
                doy = month * 30 - 15
                
                # Roll features approximates using the lag inputs
                features_row = [
                    no2_sat, so2_sat, co_sat, o3_sat, hcho_sat,
                    no2_gr, co_gr, o3_gr, so2_gr, temp, rh, wind_speed,
                    lat, lon,
                    np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
                    np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365),
                    day_of_week, season_id,
                    pm25_lag, pm25_lag, pm25_lag, # lag 1, 3, 7
                    pm25_lag, pm25_lag,           # roll 3, 7
                    aqi_lag
                ]
                
                input_arr = np.array([features_row])
                
                try:
                    # Scale inputs
                    X_scaled = scaler_X.transform(input_arr)
                    X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
                    
                    # Predict & Inverse Scale
                    y_pred_scaled = model.predict(X_lstm, verbose=0).flatten()
                    pred_aqi = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
                    pred_aqi = np.clip(pred_aqi, 0, 500)
                    
                    # Determine classification & colors
                    if pred_aqi <= 50:
                        status_class = "aqi-good"
                        status_name = "Good"
                        status_desc = "Air quality is satisfactory, and air pollution poses little or no risk."
                    elif pred_aqi <= 100:
                        status_class = "aqi-satisfactory"
                        status_name = "Satisfactory"
                        status_desc = "Air quality is acceptable; however, there may be a risk for some people."
                    elif pred_aqi <= 150:
                        status_class = "aqi-moderate"
                        status_name = "Moderate"
                        status_desc = "Members of sensitive groups may experience health effects."
                    elif pred_aqi <= 200:
                        status_class = "aqi-poor"
                        status_name = "Poor"
                        status_desc = "Everyone may begin to experience health effects."
                    elif pred_aqi <= 300:
                        status_class = "aqi-very-poor"
                        status_name = "Very Poor"
                        status_desc = "Health alert: The risk of health effects is increased for everyone."
                    else:
                        status_class = "aqi-severe"
                        status_name = "Severe"
                        status_desc = "Health warning of emergency conditions: Everyone is more likely to be affected."

                    # Result display
                    col_res1, col_res2 = st.columns([1, 2])
                    with col_res1:
                        st.markdown(f"""
                        <div class="card" style="text-align: center;">
                            <h3>Predicted AQI</h3>
                            <div class="metric-value {status_class}">{pred_aqi:.1f}</div>
                            <h4 class="{status_class}">{status_name}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_res2:
                        st.markdown(f"""
                        <div class="card" style="height: 100%;">
                            <h3>Health Guidance</h3>
                            <p style="font-size: 1.1rem; margin-top: 1rem;">{status_desc}</p>
                            <hr style="border: 1px solid rgba(255,255,255,0.05);">
                            <p style="font-size: 0.9rem; color: #a0aec0;">
                                <strong>Prediction Parameters used:</strong><br>
                                Location: {lat:.4f}°N, {lon:.4f}°E | Temp: {temp}°C | Wind: {wind_speed} m/s | Yesterday AQI: {aqi_lag}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as ex:
                    st.error(f"Error during calculation/scaling: {ex}")

# ── TAB 3: Model Performance & Insights ───────────────────────────────────────
with tab3:
    st.header("Deep Learning Model Insights")
    st.write(
        "Our system utilizes a custom hybrid **CNN-LSTM (Convolutional Neural Network - Long Short-Term Memory)** network. "
        "The CNN layers extract high-level feature interactions between satellite column densities and meteorological inputs, "
        "while the LSTM layers capture sequential, temporal lags from historical monitoring station records."
    )
    
    col_perf1, col_perf2 = st.columns(2)
    
    with col_perf1:
        st.subheader("Model Evaluation Summary")
        val_img_path = os.path.join(OUTPUT_DIR, "model_validation.png")
        dashboard_img_path = os.path.join(OUTPUT_DIR, "validation_dashboard.png")
        
        if os.path.exists(dashboard_img_path):
            st.image(dashboard_img_path, use_container_width=True, caption="Model validation metrics and performance charts")
        elif os.path.exists(val_img_path):
            st.image(val_img_path, use_container_width=True, caption="Model validation loss curves")
        else:
            st.info("Validation plot image is currently not found in the outputs directory.")
            
    with col_perf2:
        st.subheader("Features & Abstractions")
        st.markdown("""
        * **Remote Sensing**: TROPOMI data on TROPOMI Sentinel-5P platform ($NO_2$, $SO_2$, $CO$, $O_3$, $HCHO$) are fed into spatial interpolation layers.
        * **Climatology**: Temperature, Relative Humidity, Wind Speed, and Wind Direction ground readings.
        * **Sequential Lag Features**: Rolling 3-day and 7-day average levels of $PM_{2.5}$ particles are calculated along with yesterday's local station readings to leverage temporal autocorrelation in air patterns.
        * **Temporal Encodings**: Trigonometric transformations ($sin$/$cos$) are applied to Month and Day of Year fields to avoid discontinuous boundaries at year turns and model seasonal patterns smoothly.
        """)
        
        # Add architecture block diagram/text
        st.code("""
  Input Vector (26 Features)
             │
             ▼
  1D CNN Layer (Feature Extraction)
             │
             ▼
  LSTM Layer (Temporal/Sequence Modeling)
             │
             ▼
  Dense Layer (Fully Connected)
             │
             ▼
  Linear Activation → Output Surface AQI [0, 500]
        """, language="text")

# ── TAB 4: Custom Cross-Verification ──────────────────────────────────────────
with tab4:
    st.header("Upload Validation Data for Crosscheck")
    st.write(
        "Upload a CSV file containing validation results for another year or a test run. "
        "The system will automatically run our 5-point data verification checks and generate accuracy charts."
    )
    
    # Show format sample
    st.markdown("""
    **Required CSV Columns:**
    * `station` (Name of the monitoring station)
    * `actual_AQI` (Actual observed AQI)
    * `predicted_AQI` (Predicted AQI by the model)
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            val_df = pd.read_csv(uploaded_file)
            
            # Check for required columns
            required_cols = ["station", "actual_AQI", "predicted_AQI"]
            missing_cols = [c for c in required_cols if c not in val_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                st.success("File uploaded successfully! Running cross-checks...")
                
                # Check for NaN and drop/fill
                val_df = val_df.dropna(subset=required_cols)
                
                # R2 and RMSE
                r2 = r2_score(val_df["actual_AQI"], val_df["predicted_AQI"])
                rmse = np.sqrt(mean_squared_error(val_df["actual_AQI"], val_df["predicted_AQI"]))
                
                # --- Checks layout ---
                col_c1, col_c2 = st.columns(2)
                
                with col_c1:
                    st.subheader("📋 5-Point Sanity Checks")
                    
                    # Check 1: Range Validity
                    st.write("**[Check 1] Prediction range validity**")
                    min_act, max_act = val_df['actual_AQI'].min(), val_df['actual_AQI'].max()
                    min_pred, max_pred = val_df['predicted_AQI'].min(), val_df['predicted_AQI'].max()
                    st.write(f"Actual AQI Range: {min_act:.1f} to {max_act:.1f}")
                    st.write(f"Predicted AQI Range: {min_pred:.1f} to {max_pred:.1f}")
                    out_of_range = ((val_df["predicted_AQI"] < 0) | (val_df["predicted_AQI"] > 500)).sum()
                    if out_of_range == 0:
                        st.success("✓ All predictions are within valid range (0-500).")
                    else:
                        st.warning(f"⚠ {out_of_range} predictions are out of the valid 0-500 range.")
                        
                    # Check 2: R2 Sanity
                    st.write("**[Check 2] R² Sanity check**")
                    st.write(f"R² Score: **{r2:.4f}** | RMSE: **{rmse:.2f}**")
                    if r2 > 0.95:
                        st.warning("⚠ R² > 0.95: Possible overfitting, check train/test leakage.")
                    elif r2 > 0.85:
                        st.success("✓ R² is excellent — model is highly reliable.")
                    elif r2 > 0.70:
                        st.info("~ R² is acceptable — model is usable.")
                    else:
                        st.error("✗ R² too low — model needs improvement.")
                        
                    # Check 4: Error distribution
                    st.write("**[Check 4] Error distribution**")
                    errors = val_df["actual_AQI"] - val_df["predicted_AQI"]
                    within_20 = (np.abs(errors) <= 20).mean() * 100
                    within_50 = (np.abs(errors) <= 50).mean() * 100
                    within_100 = (np.abs(errors) <= 100).mean() * 100
                    st.write(f"* Within ±20 AQI: **{within_20:.1f}%**")
                    st.write(f"* Within ±50 AQI: **{within_50:.1f}%**")
                    st.write(f"* Within ±100 AQI: **{within_100:.1f}%**")
                    if within_50 > 80:
                        st.success("✓ Good: 80%+ predictions are within ±50 AQI.")
                    else:
                        st.warning("⚠ Below 80% within ±50 AQI: model may need tuning.")

                with col_c2:
                    st.subheader("🏫 Per Station Comparison")
                    station_data = []
                    for station in sorted(val_df["station"].unique()):
                        s = val_df[val_df["station"] == station]
                        actual_mean = s["actual_AQI"].mean()
                        pred_mean = s["predicted_AQI"].mean()
                        diff = pred_mean - actual_mean
                        status = "✓" if abs(diff) < 30 else "⚠"
                        station_data.append({
                            "Station": station,
                            "Actual Mean": round(actual_mean, 1),
                            "Predicted Mean": round(pred_mean, 1),
                            "Difference": round(diff, 1),
                            "Status": status
                        })
                    st.dataframe(pd.DataFrame(station_data), use_container_width=True)
                    
                    # Reality check
                    st.write("**[Check 5] Reality check vs expected India ranges**")
                    known_ranges = {
                        "Narela_Delhi": (150, 450), "Amritsar": (80, 300), "Lucknow": (100, 350),
                        "Mumbai": (50, 180), "Bengaluru": (30, 120), "Kolkata": (80, 300),
                        "Jaipur": (60, 250), "Gurugram": (100, 400), "Chandigarh": (80, 280),
                    }
                    rc_items = []
                    for station, (lo, hi) in known_ranges.items():
                        s = val_df[val_df["station"] == station]
                        if len(s) == 0:
                            continue
                        pred_mean = s["predicted_AQI"].mean()
                        if lo <= pred_mean <= hi:
                            rc_items.append(f"✓ **{station}**: Predicted {pred_mean:.0f} (Expected {lo}-{hi})")
                        else:
                            rc_items.append(f"✗ **{station}**: Predicted {pred_mean:.0f} (Expected {lo}-{hi}) ⚠")
                    st.markdown("\n".join(f"* {item}" for item in rc_items))

                # --- Plots ---
                st.markdown("---")
                st.subheader("📈 Validation Plots")
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                # Scatter
                axes[0].scatter(val_df["actual_AQI"], val_df["predicted_AQI"], alpha=0.5, s=15, color="#1fa2ff")
                axes[0].plot([0, 500], [0, 500], "r--", lw=2, label="Perfect fit")
                axes[0].plot([0, 500], [50, 550], "g--", lw=1, alpha=0.5, label="+50 error band")
                axes[0].plot([0, 500], [-50, 450], "g--", lw=1, alpha=0.5)
                axes[0].set_xlim(0, 500)
                axes[0].set_ylim(0, 500)
                axes[0].set_xlabel("Actual AQI")
                axes[0].set_ylabel("Predicted AQI")
                axes[0].set_title(f"Scatter — R²={r2:.3f} | RMSE={rmse:.1f}")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Error histogram
                axes[1].hist(errors, bins=30, color="#12d6df", edgecolor="white", alpha=0.8)
                axes[1].axvline(0, color="red", lw=2, label="Zero error")
                axes[1].axvline(50, color="orange", lw=1.5, linestyle="--", label="±50 band")
                axes[1].axvline(-50, color="orange", lw=1.5, linestyle="--")
                axes[1].set_xlabel("Prediction Error (Actual - Predicted)")
                axes[1].set_ylabel("Count")
                axes[1].set_title(f"Error Distribution\n{within_50:.1f}% within ±50 AQI")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.suptitle("Uploaded Data Verification Report", fontsize=14, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #a0aec0; font-size: 0.8rem;'>AQI Deep Learning Platform © 2026. Made with Streamlit.</p>", 
    unsafe_allow_html=True
)
