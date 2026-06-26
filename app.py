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

# ── Sidebar Improvements ──────────────────────────────────────────────────────
st.sidebar.title("🌍 AQI Predictor Control Panel")

# 1. Model Status Indicator
if model is not None and error_msg is None:
    st.sidebar.success("🟢 Model Loaded — R²=0.901")
else:
    st.sidebar.error("🔴 Model Not Loaded")
    if error_msg:
        st.sidebar.code(error_msg, language="text")

# 2. Current AQI Status Info Box (January warning)
st.sidebar.markdown("""
<div class="card" style="padding: 1rem; border-color: rgba(255, 78, 0, 0.3);">
    <h4 style="margin-top: 0; color: #ff7e00;">⚠️ Current AQI Status</h4>
    <p style="font-size: 0.9rem; margin-bottom: 0;">
        <strong>January</strong> represents the peak winter pollution season. 
        Temperature inversions and low winds trap particulate matter, causing 
        consistently higher AQI readings (often Poor to Severe) across the 
        Indo-Gangetic Plain.
    </p>
</div>
""", unsafe_allow_html=True)

# 3. Project Info Box
st.sidebar.markdown("""
<div class="card" style="padding: 1rem; margin-top: 1rem;">
    <h4 style="margin-top: 0; color: #1fa2ff;">📋 Project Metadata</h4>
    <p style="font-size: 0.85rem; line-height: 1.4; margin-bottom: 0;">
        <strong>ISRO Hackathon</strong> — Objective 1<br>
        <strong>Model:</strong> Deep Learning CNN-LSTM<br>
        <strong>Training Period:</strong> Jan - Mar 2025<br>
        <strong>Stations Used:</strong> 9 Indian Cities<br>
        <strong>Input Features:</strong> 26 columns
    </p>
</div>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Air Quality Index (AQI) Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Spatial Mapping and Real-time Deep Learning Predictions over India using CNN-LSTM & Satellite Data</div>', unsafe_allow_html=True)

if error_msg:
    st.error(f"Error loading machine learning assets from '{MODEL_DIR}': {error_msg}")
    st.warning("Please make sure the models folder containing 'cnn_lstm_best.h5', 'scaler_X.pkl', and 'scaler_y.pkl' is uploaded correctly to your repository.")

# ── Tabs Configuration ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Spatial AQI Maps (India)", 
    "🔮 Real-Time AQI Calculator", 
    "📊 Model Performance & Insights",
    "🔍 Custom Cross-Verification",
    "ℹ️ About This Project"
])

# ── TAB 1: Spatial AQI Maps (India) ───────────────────────────────────────────
with tab1:
    st.header("Spatial AQI Mapping over India (Jan - Mar 2025)")
    st.write(
        "These maps visualize predicted surface AQI across India, generated using our CNN-LSTM deep learning model combining TROPOMI satellite columns and ground-based meteorological/station data."
    )
    
    # 1. Clean Dropdown Selectbox instead of Radio buttons
    month_selection = st.selectbox(
        "Select Prediction Month:",
        ["January 2025", "February 2025", "March 2025"]
    )
    
    # Map map names to pre-generated files
    map_files = {
        "January 2025": "aqi_map_jan2025.png",
        "February 2025": "aqi_map_feb2025.png",
        "March 2025": "aqi_map_mar2025.png"
    }
    
    filename = map_files[month_selection]
    map_path = os.path.join(OUTPUT_DIR, filename)
    
    # Metrics Row
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("January Mean AQI", "126", help="High pollution winter month")
    with m_col2:
        st.metric("February Mean AQI", "88", help="Transition month showing improvement")
    with m_col3:
        st.metric("March Mean AQI", "67", help="Spring month showing clean air quality")
        
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # st.spinner when loading map
        with st.spinner("Loading satellite AQI predictions..."):
            if os.path.exists(map_path):
                st.image(map_path, use_container_width=True, 
                         caption=f"Figure 1: Spatial surface AQI distribution predicted over India for {month_selection} based on CNN-LSTM model.")
            else:
                st.warning(f"Map file '{filename}' not found in the 'outputs/' folder. Please ensure output images are uploaded.")
            
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

    st.markdown("---")
    st.subheader("🗓️ Seasonal Comparison (January - March 2025)")
    st.write("Compare the spatial maps side-by-side to observe the clean-air transition pattern across India:")
    
    # Side-by-side comparison row
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        jan_path = os.path.join(OUTPUT_DIR, "aqi_map_jan2025.png")
        if os.path.exists(jan_path):
            st.image(jan_path, use_container_width=True, caption="January 2025 Map")
        else:
            st.error("Jan Map missing")
        st.markdown("<div style='text-align: center; font-weight: bold;'>January Mean AQI: 126</div>", unsafe_allow_html=True)
        # progress bar showing Mean AQI as percentage of 500 maximum
        st.progress(126 / 500)
        
    with col_c2:
        feb_path = os.path.join(OUTPUT_DIR, "aqi_map_feb2025.png")
        if os.path.exists(feb_path):
            st.image(feb_path, use_container_width=True, caption="February 2025 Map")
        else:
            st.error("Feb Map missing")
        st.markdown("<div style='text-align: center; font-weight: bold;'>February Mean AQI: 88</div>", unsafe_allow_html=True)
        st.progress(88 / 500)
        
    with col_c3:
        mar_path = os.path.join(OUTPUT_DIR, "aqi_map_mar2025.png")
        if os.path.exists(mar_path):
            st.image(mar_path, use_container_width=True, caption="March 2025 Map")
        else:
            st.error("Mar Map missing")
        st.markdown("<div style='text-align: center; font-weight: bold;'>March Mean AQI: 67</div>", unsafe_allow_html=True)
        st.progress(67 / 500)
        
    # Key Observations Expandable Box
    with st.expander("🔍 Key Observations", expanded=True):
        st.markdown("""
        * **January 2025:** Mean AQI 126 — IGP belt (Delhi-Lucknow) shows Orange/Poor
        * **February 2025:** Mean AQI 88 — Mostly Yellow/Satisfactory
        * **March 2025:** Mean AQI 67 — Mostly Green/Good
        * **Seasonal pattern confirmed:** pollution decreases Jan → Feb → Mar
        * **Indo-Gangetic Plain consistently most polluted region**
        """)

# ── TAB 2: Real-Time AQI Calculator ───────────────────────────────────────────
with tab2:
    st.header("Real-Time AQI Deep Learning Calculator")
    st.write(
        "Select a monitoring station or specify meteorological and satellite values below to calculate the predicted AQI in real-time using our trained Keras CNN-LSTM model."
    )
    
    if ref_df is not None:
        # User can pick a station to load defaults
        unique_stations = sorted(ref_df["station"].unique())
        # Fixed dropdown default to Narela_Delhi
        default_idx = unique_stations.index("Narela_Delhi") if "Narela_Delhi" in unique_stations else 0
        
        selected_station = st.selectbox(
            "Quick Select Monitoring Station (loads defaults):",
            unique_stations,
            index=default_idx
        )
        
        # Get defaults from selected station
        station_df = ref_df[ref_df["station"] == selected_station]
        # Get mean values for the station
        defaults = station_df.mean(numeric_only=True).to_dict()
        defaults["lat"] = station_df["lat"].iloc[0] if "lat" in station_df.columns else 28.856
        defaults["lon"] = station_df["lon"].iloc[0] if "lon" in station_df.columns else 77.101
    else:
        st.info("Reference dataset not found at `data/processed/processed_dataset.csv`. Using default regional values.")
        defaults = {
            "lat": 28.856, "lon": 77.101,
            "temp": 15.0, "rh": 65.0, "wind_speed": 1.5,
            "NO2_sat": 5e-5, "SO2_sat": 2e-4, "CO_sat": 4e-2, "O3_sat": 1.2e-1, "HCHO_sat": 1.5e-4,
            "NO2": 30.0, "CO": 1.0, "O3": 50.0, "SO2": 10.0,
            "PM25_lag1": 80.0, "AQI_lag1": 150.0
        }

    # Setup columns for parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌐 Location & Time")
        lat = st.number_input("Latitude (°N)", min_value=8.0, max_value=38.0, value=float(defaults.get("lat", 28.856)), step=0.01)
        lon = st.number_input("Longitude (°E)", min_value=68.0, max_value=98.0, value=float(defaults.get("lon", 77.101)), step=0.01)
        
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
                # Approximate day of year from month
                doy = month * 30 - 15
                
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
                        status_color = "#00e400"
                        status_desc = "Air quality is satisfactory, and air pollution poses little or no risk."
                        health_rec = "🌳 **Outdoor Activity:** Safe to be outside.\n😷 **Precautions:** No specific precautions needed.\n⚠️ **Sensitive Groups:** No warnings."
                    elif pred_aqi <= 100:
                        status_class = "aqi-satisfactory"
                        status_name = "Satisfactory"
                        status_color = "#92d050"
                        status_desc = "Air quality is acceptable; however, there may be a risk for some people."
                        health_rec = "🌳 **Outdoor Activity:** Safe to be outside.\n😷 **Precautions:** Sensitive people may need minor masks if symptomatic.\n⚠️ **Sensitive Groups:** Minor breathing discomfort to exceptionally sensitive individuals."
                    elif pred_aqi <= 150:
                        status_class = "aqi-moderate"
                        status_name = "Moderate"
                        status_color = "#ffff00"
                        status_desc = "Members of sensitive groups may experience health effects."
                        health_rec = "🌳 **Outdoor Activity:** Reduce heavy outdoor activity.\n😷 **Precautions:** Consider wearing masks if outside for long periods.\n⚠️ **Sensitive Groups:** Asthma, heart, and lung disease patients should avoid heavy outdoor exertion."
                    elif pred_aqi <= 200:
                        status_class = "aqi-poor"
                        status_name = "Poor"
                        status_color = "#ff7e00"
                        status_desc = "Everyone may begin to experience health effects."
                        health_rec = "🌳 **Outdoor Activity:** Limit prolonged outdoor activities.\n😷 **Precautions:** Wear standard masks when stepping outside.\n⚠️ **Sensitive Groups:** Sensitive groups should remain indoors."
                    elif pred_aqi <= 300:
                        status_class = "aqi-very-poor"
                        status_name = "Very Poor"
                        status_color = "#ff0000"
                        status_desc = "Health alert: The risk of health effects is increased for everyone."
                        health_rec = "🌳 **Outdoor Activity:** Avoid prolonged outdoor exertion. Stay indoors.\n😷 **Precautions:** Use N95 masks if outdoor travel is mandatory.\n⚠️ **Sensitive Groups:** High risk. Severe symptoms possible for respiratory conditions."
                    else:
                        status_class = "aqi-severe"
                        status_name = "Severe"
                        status_color = "#8f3f97"
                        status_desc = "Health warning of emergency conditions: Everyone is more likely to be affected."
                        health_rec = "🌳 **Outdoor Activity:** Strictly avoid all outdoor exertion. Lock windows/doors.\n😷 **Precautions:** N95 masks mandatory. Use indoor air purifiers.\n⚠️ **Sensitive Groups:** Serious health risk. Hospitalizations possible."

                    # Result display layout
                    col_res1, col_res2 = st.columns([1, 2])
                    
                    with col_res1:
                        st.markdown(f"""
                        <div class="card" style="text-align: center;">
                            <h3>Predicted AQI</h3>
                            <div class="metric-value {status_class}">{pred_aqi:.1f}</div>
                            <h4 class="{status_class}">{status_name}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Gauge Chart using Matplotlib
                        fig, ax = plt.subplots(figsize=(6, 1.4))
                        colors = ["#00e400", "#92d050", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
                        bounds = [0, 50, 100, 150, 200, 300, 400, 500]
                        for i in range(len(colors)):
                            ax.barh(0, bounds[i+1]-bounds[i], left=bounds[i], height=0.4, color=colors[i], align='center', alpha=0.9)
                        ax.axvline(pred_aqi, color='black', lw=3.0, ymin=0.1, ymax=0.9)
                        ax.plot(pred_aqi, 0, 'k^', markersize=10)
                        ax.text(pred_aqi, 0.35, f"{pred_aqi:.1f}", color='black', fontweight='bold', fontsize=10, ha='center',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", alpha=0.9))
                        ax.set_xlim(0, 500)
                        ax.set_ylim(-0.4, 0.7)
                        ax.set_xticks(bounds)
                        ax.set_xticklabels(["0", "50\nGood", "100\nSat.", "150\nMod.", "200\nPoor", "300\nV.Poor", "400\nSev.", "500"], fontsize=7)
                        ax.set_yticks([])
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    with col_res2:
                        st.markdown(f"""
                        <div class="card" style="height: 100%;">
                            <h3>Health Guidance</h3>
                            <p style="font-size: 1.1rem; margin-top: 1rem; font-weight: 600;">{status_desc}</p>
                            <div style="background: rgba(255,255,255,0.02); padding: 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin: 0.5rem 0;">
                                {health_rec}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Comparison Box vs Typical Historical Range
                    st.markdown("### 📊 Climatological Context")
                    if ref_df is not None and 'station' in ref_df.columns and 'season' in ref_df.columns:
                        hist_sub = ref_df[(ref_df["station"] == selected_station) & (ref_df["season"] == season_id)]
                        if not hist_sub.empty:
                            min_h = hist_sub["AQI"].min()
                            max_h = hist_sub["AQI"].max()
                            mean_h = hist_sub["AQI"].mean()
                            
                            st.info(f"**Typical Historical Range for {selected_station} in {season}:** "
                                    f"**{min_h:.1f}** to **{max_h:.1f}** AQI (Average: **{mean_h:.1f}** AQI). "
                                    f"Your predicted AQI of **{pred_aqi:.1f}** is "
                                    f"{'higher than typical' if pred_aqi > mean_h else 'within or below typical'} conditions.")
                        else:
                            st.info(f"No specific historical data available for {selected_station} during the {season} season.")
                    else:
                        st.info("Historical data comparison details not available.")
                        
                except Exception as ex:
                    st.error(f"Error during calculation/scaling: {ex}")

# ── TAB 3: Model Performance & Insights ───────────────────────────────────────
with tab3:
    st.header("Deep Learning Model Performance Dashboard")
    st.write(
        "Our system utilizes a custom hybrid **CNN-LSTM (Convolutional Neural Network - Long Short-Term Memory)** network. "
        "The CNN layers extract high-level feature interactions between satellite column densities and meteorological inputs, "
        "while the LSTM layers capture sequential, temporal lags from historical monitoring station records."
    )
    
    # New Metrics Row
    pm_col1, pm_col2, pm_col3, pm_col4, pm_col5 = st.columns(5)
    with pm_col1:
        st.metric("Overall R² Score", "0.901", help="Coefficient of determination")
    with pm_col2:
        st.metric("Overall RMSE", "35.8", help="Root Mean Squared Error")
    with pm_col3:
        st.metric("Overall MAE", "23.0", help="Mean Absolute Error")
    with pm_col4:
        st.metric("Pearson Correlation", "0.950", help="Correlation coefficient")
    with pm_col5:
        st.metric("Overall Bias", "-3.5", help="Mean prediction bias")

    # Main dashboard curves
    col_perf1, col_perf2 = st.columns(2)
    
    with col_perf1:
        st.subheader("Model Evaluation Summary")
        dashboard_img_path = os.path.join(OUTPUT_DIR, "validation_dashboard.png")
        if os.path.exists(dashboard_img_path):
            st.image(dashboard_img_path, use_container_width=True, 
                     caption="Figure 2: Validation scatter plot showing correlation and overall error metrics.")
        else:
            st.info("Validation dashboard plot not found.")
            
    with col_perf2:
        st.subheader("Training Loss Curves")
        val_img_path = os.path.join(OUTPUT_DIR, "model_validation.png")
        if os.path.exists(val_img_path):
            st.image(val_img_path, use_container_width=True, 
                     caption="Figure 3: Training and validation loss curves (Huber loss) showing clean convergence.")
        else:
            st.info("Validation curve image not found.")

    # More validation charts
    st.markdown("---")
    col_cperf1, col_cperf2 = st.columns(2)
    
    with col_cperf1:
        st.subheader("Temporal Time Series Trends")
        ts_img_path = os.path.join(OUTPUT_DIR, "timeseries_all_stations.png")
        if os.path.exists(ts_img_path):
            st.image(ts_img_path, use_container_width=True, 
                     caption="Figure 4: Predicted vs actual AQI trends across multiple stations over time.")
        else:
            st.info("Timeseries plot not found.")
            
    with col_cperf2:
        st.subheader("Station-wise Performance Scatter")
        stat_img_path = os.path.join(OUTPUT_DIR, "validation_per_station.png")
        if os.path.exists(stat_img_path):
            st.image(stat_img_path, use_container_width=True, 
                     caption="Figure 5: Station-wise prediction accuracy showing robust fit across cities.")
        else:
            st.info("Station validation plot not found.")

    # Station and Seasonal Tables
    st.markdown("---")
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.subheader("🏫 Station-wise Validation Details")
        st_path = os.path.join(OUTPUT_DIR, "validation_per_station.csv")
        if os.path.exists(st_path):
            df_st = pd.read_csv(st_path)
            
            # Color code rules
            def style_r2(val):
                try:
                    val_float = float(val)
                    if val_float > 0.85:
                        return 'background-color: rgba(0, 228, 0, 0.25); color: #00e400; font-weight: bold;'
                    elif val_float >= 0.70:
                        return 'background-color: rgba(255, 126, 0, 0.25); color: #ff7e00; font-weight: bold;'
                    else:
                        return 'background-color: rgba(255, 0, 0, 0.25); color: #ff0000; font-weight: bold;'
                except:
                    return ''
            
            st.dataframe(df_st.style.applymap(style_r2, subset=["R2"]), use_container_width=True)
        else:
            st.error("validation_per_station.csv missing")
            
    with col_t2:
        st.subheader("🍂 Seasonal Validation Details")
        seas_path = os.path.join(OUTPUT_DIR, "validation_per_season.csv")
        if os.path.exists(seas_path):
            df_seas = pd.read_csv(seas_path)
            st.dataframe(df_seas, use_container_width=True)
        else:
            st.error("validation_per_season.csv missing")

    # Model Architecture Expandable Box
    with st.expander("🛠️ Model Architecture & Hyperparameters", expanded=False):
        st.markdown("""
        **CNN-LSTM Neural Network Layers:**
        * **Input Layer:** 26 temporal & spatial features
        * **Conv1D Layer 1:** 128 filters → Batch Normalization → ReLU
        * **Conv1D Layer 2:** 256 filters → Batch Normalization → ReLU → Dropout (Rate: 0.2)
        * **LSTM Layer 1:** 256 cells (return sequences: True)
        * **LSTM Layer 2:** 128 cells (return sequences: False) → Dropout (Rate: 0.3)
        * **Dense Layer 1:** 128 units → Batch Normalization
        * **Dense Layer 2:** 64 units → Dropout (Rate: 0.2)
        * **Output Layer:** Dense 1 unit (Linear activation → final AQI bounded [0, 500])
        
        **Hyperparameters:**
        * **Loss Function:** Huber Loss (robust to outliers)
        * **Optimizer:** Adam (learning rate = 0.0005)
        * **Batch Size:** 16
        """)

# ── TAB 4: Custom Cross-Verification ──────────────────────────────────────────
with tab4:
    st.header("Upload or Load Validation Data for Crosscheck")
    st.write(
        "Upload a CSV file containing validation results for another year or a test run, "
        "or load our standard project validation results to run the 5-point sanity checks automatically."
    )
    
    # Required Format
    st.markdown("""
    **Required CSV Columns:**
    * `station` (Name of the monitoring station)
    * `actual_AQI` (Actual observed AQI)
    * `predicted_AQI` (Predicted AQI by the model)
    """)
    
    # Button to Load preloaded data
    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        load_preloaded = st.button("📂 Load our validation results", use_container_width=True)
        
    uploaded_file = st.file_uploader("Or choose a custom CSV file", type="csv")
    
    val_df = None
    if load_preloaded:
        pre_path = os.path.join(OUTPUT_DIR, "validation_results.csv")
        if os.path.exists(pre_path):
            val_df = pd.read_csv(pre_path)
            st.success("Successfully loaded standard validation results!")
        else:
            st.error("validation_results.csv file was not found in outputs/")
    elif uploaded_file is not None:
        try:
            val_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            
    if val_df is not None:
        try:
            required_cols = ["station", "actual_AQI", "predicted_AQI"]
            missing_cols = [c for c in required_cols if c not in val_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
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
                        
                    # Check 3: Per station mean comparison table (Fixed: was requested in guidelines)
                    st.write("**[Check 3] Per station mean comparison**")
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

                with col_c2:
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
                        
                    # Check 5: Reality check
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
                
                plt.suptitle("Data Verification Report", fontsize=14, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error running validation: {e}")

# ── TAB 5: About This Project ─────────────────────────────────────────────────
with tab5:
    st.header("About — ISRO AQI Hackathon 2025")
    
    # Section 1: Problem Statement
    st.subheader("📌 Problem Statement")
    st.write(
        "Most Indians live far from physical air quality monitors, leading to vast spatial data gaps. "
        "This project uses ISRO INSAT-3D and ESA Sentinel-5P satellite data with a CNN-LSTM deep learning model "
        "to predict surface Air Quality Index (AQI) at every 0.25° grid point across India — replacing the need "
        "for expensive physical ground sensors."
    )
    
    # Section 2: Data Sources Table
    st.subheader("📅 Data Sources")
    st.markdown("""
    | Source | Data | Spatial Resolution | Period |
    | :--- | :--- | :--- | :--- |
    | **TROPOMI Sentinel-5P** | NO2, SO2, CO, O3, HCHO column densities | 10 km | 2025 |
    | **CPCB Ground Stations** | PM2.5, NO2, CO, O3, SO2, meteorological context | Point location | 2025 |
    | **ERA5 Reanalysis** | Temperature, Relative Humidity, Wind Speed | 0.1° | 2025 |
    """)
    
    # Section 3: Methodology (numbered)
    st.subheader("🔄 Methodology Pipeline")
    st.markdown("""
    1. **Download TROPOMI Satellite Data**: Acquire NO2, SO2, CO, O3, and HCHO columns from Google Earth Engine.
    2. **Collect Ground Measurements**: Download reference ground AQI parameters from 9 CPCB monitoring stations.
    3. **Spatial Interpolation & Alignment**: Apply linear interpolation to map satellite columns directly to the monitoring station coordinates.
    4. **Feature Engineering**: Generate temporal lags (1, 3, 7 days), rolling averages, and sin/cos encodings for Month and Day of Year.
    5. **Train CNN-LSTM Deep Learning Model**: Process combined satellite + meteorological features to estimate ground surface AQI.
    6. **Validate Station Accuracy**: Perform cross-validation and compute errors against held-out station test data.
    7. **Predict AQI on 0.25° India Grid**: Deploy the trained network model to generate continuous spatial AQI maps over India.
    """)
    
    # Section 4: Stations Used
    st.subheader("🏫 Ground Stations Utilized")
    st.write("The deep learning model was trained and validated using continuous data from these 9 geographical stations:")
    
    stations_data = [
        {"City": "Delhi (Narela)", "State": "Delhi", "Latitude": "28.8560° N", "Longitude": "77.1010° E"},
        {"City": "Amritsar", "State": "Punjab", "Latitude": "31.6340° N", "Longitude": "74.8723° E"},
        {"City": "Chandigarh", "State": "Chandigarh", "Latitude": "30.7333° N", "Longitude": "76.7794° E"},
        {"City": "Gurugram", "State": "Haryana", "Latitude": "28.4595° N", "Longitude": "77.0266° E"},
        {"City": "Jaipur", "State": "Rajasthan", "Latitude": "26.9124° N", "Longitude": "75.7873° E"},
        {"City": "Kolkata", "State": "West Bengal", "Latitude": "22.5154° N", "Longitude": "88.3632° E"},
        {"City": "Lucknow", "State": "Uttar Pradesh", "Latitude": "26.8467° N", "Longitude": "80.9462° E"},
        {"City": "Mumbai", "State": "Maharashtra", "Latitude": "19.2307° N", "Longitude": "72.8567° E"},
        {"City": "Bengaluru", "State": "Karnataka", "Latitude": "12.9121° N", "Longitude": "77.6097° E"},
    ]
    st.table(pd.DataFrame(stations_data))
    
    # Section 5: Relevance to National Priorities
    st.subheader("🚀 Relevance to National Priorities")
    st.markdown("""
    * **National Clean Air Programme (NCAP):** Supports tracking surface pollution and spatial patterns in regions without monitor coverage.
    * **United Nations SDG 11.6:** Helps reduce the adverse environmental impact of cities by improving air quality monitoring structures.
    * **ISRO INSAT-3D Integration:** Demonstrates integration potentials between satellite remote-sensing products and ground CPCB data.
    * **Chintan Shivir 2.0:** Aligns with technology modernization plans for environmental monitoring and data-driven policymaking.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #a0aec0; font-size: 0.8rem;'>AQI Deep Learning Platform — ISRO Hackathon 2025. Made with Streamlit.</p>", 
    unsafe_allow_html=True
)
