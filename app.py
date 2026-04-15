import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# --- Page Configuration ---
st.set_page_config(page_title="AeroGuard: Engine RUL Predictor", layout="wide")

st.title("✈️ AeroGuard: Predictive Maintenance Dashboard")
st.markdown("""
This dashboard predicts the **Remaining Useful Life (RUL)** of aircraft engines using the NASA C-MAPSS dataset.
It monitors sensor telemetry to detect mechanical degradation before failure occurs.
""")

st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4259;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Load the Model ---
MODEL_PATH = 'rul_model.joblib'

@st.cache_resource
def load_prediction_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_prediction_model()

def get_importance(model, feature_names):
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    return feat_imp

# --- 2. Data Loading & Labeling ---
@st.cache_data
def load_data():
    # Define Column Names
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['set1', 'set2', 'set3']
    sensor_names = [f's_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    # Load Training Data
    df = pd.read_csv('CMAPSSData/train_FD001.txt', sep='\s+', header=None, names=col_names)
    
    # Calculate Ground Truth RUL (for comparison)
    df['RUL_actual'] = df.groupby('unit_nr')['time_cycles'].transform('max') - df['time_cycles']
    return df

data = load_data()

# --- 3. Sidebar Selection ---
st.sidebar.header("Engine Selection")
engine_id = st.sidebar.selectbox("Select Engine Unit ID", data['unit_nr'].unique())

# Filter data for the selected engine
engine_data = data[data['unit_nr'] == engine_id].reset_index(drop=True)

# --- 4. Prediction Logic ---
# These are the 14 features we used in 'train_model.py'
features = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

# Get the most recent telemetry from this engine
latest_telemetry = engine_data[features].iloc[-1:]

if model:
    predicted_rul = model.predict(latest_telemetry)[0]
    actual_rul = engine_data['RUL_actual'].iloc[-1]
    
    # --- Sidebar Metrics ---
    st.sidebar.markdown("---")
    st.sidebar.metric(label="Predicted RUL", value=f"{int(predicted_rul)} Cycles", delta=f"{int(predicted_rul - actual_rul)} vs Actual")
    
    # Progress Bar (based on prediction)
    total_life_est = engine_data['time_cycles'].iloc[-1] + predicted_rul
    progress = int((engine_data['time_cycles'].iloc[-1] / total_life_est) * 100)
    
    st.sidebar.write(f"Estimated Engine Life Consumed: {progress}%")
    st.sidebar.progress(progress)

    # Alerts
    if predicted_rul < 30:
        st.sidebar.error("⚠️ CRITICAL: Maintenance Required Immediately")
    elif predicted_rul < 70:
        st.sidebar.warning("⚡ WARNING: Schedule Inspection Soon")
    else:
        st.sidebar.success("✅ Engine Status: Healthy")
else:
    st.sidebar.error("Model file not found! Please run 'train_model.py' first.")

# --- 5. Visualizing Sensor Trends ---
st.subheader(f"Live Telemetry Trends: Engine Unit {engine_id}")

# Key sensors for FD001 degradation
useful_sensors = ['s_2', 's_7', 's_11', 's_12']
sensor_labels = {
    's_2': 'Total Temp (LPT Outlet)',
    's_7': 'Total Temp (HPC Outlet)',
    's_11': 'Static Pressure (HPC Outlet)',
    's_12': 'Ratio of Fuel Flow to Static Pressure'
}

cols = st.columns(2)
for i, sensor in enumerate(useful_sensors):
    with cols[i % 2]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(engine_data['time_cycles'], engine_data[sensor], color='#2ecc71' if i > 1 else '#e74c3c', linewidth=1.5)
        ax.set_title(sensor_labels[sensor], fontsize=12)
        ax.set_xlabel("Flight Cycles")
        ax.set_ylabel("Reading")
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)

st.write("---")
col_a, col_b = st.columns([1, 2])

with col_a:
    st.subheader("Key Risk Factors")
    st.info("These sensors are contributing most to the current RUL prediction.")
    
    # Get importance and plot
    feat_imp = get_importance(model, features)
    fig_imp, ax_imp = plt.subplots()
    feat_imp.tail(7).plot(kind='barh', ax=ax_imp, color='skyblue') # Top 7 sensors
    ax_imp.set_title("Sensor Impact on Prediction")
    st.pyplot(fig_imp)

with col_b:
    st.subheader("Maintenance Recommendation")
    if predicted_rul < 30:
        st.write(f"**Primary Concern:** High variance detected in `{feat_imp.idxmax()}`.")
        st.write("👉 **Action:** Schedule immediate borescope inspection of the High-Pressure Compressor (HPC).")
    else:
        st.write("👉 **Action:** Continue standard flight operations. Next automated check in 10 cycles.")

# --- 6. Raw Data View ---
with st.expander("View Raw Telemetry Log"):
    st.dataframe(engine_data.tail(10))

st.write("---")
st.subheader("RUL History & Prediction Accuracy")

# Calculate prediction for all cycles of this engine for the chart
# (In a real app, you'd pre-calculate this to save speed)
history_preds = model.predict(engine_data[features])

fig_trend, ax_trend = plt.subplots(figsize=(12, 4))
ax_trend.plot(engine_data['time_cycles'], engine_data['RUL_actual'], label='Actual RUL', color='white', linestyle='--', alpha=0.6)
ax_trend.plot(engine_data['time_cycles'], history_preds, label='ML Predicted RUL', color='#00d1ff', linewidth=2)
ax_trend.set_ylabel("Remaining Cycles")
ax_trend.set_xlabel("Flight Cycles")
ax_trend.legend()
ax_trend.grid(True, alpha=0.2)
st.pyplot(fig_trend)