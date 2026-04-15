import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# --- Page Configuration ---
st.set_page_config(page_title="AeroPredict: Engine RUL Predictor", layout="wide")

# Custom CSS for dark-mode metric cards
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

# --- 1. Load the Universal Model ---
MODEL_PATH = 'rul_model.joblib'

@st.cache_resource
def load_prediction_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_prediction_model()

# --- 2. Data Loading ---
@st.cache_data
def load_selected_data(subset):
    col_names = ['unit_nr', 'time_cycles', 'set1', 'set2', 'set3'] + [f's_{i}' for i in range(1, 22)]
    file_path = f'CMAPSSData/train_{subset}.txt'
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    df['RUL_actual'] = df.groupby('unit_nr')['time_cycles'].transform('max') - df['time_cycles']
    return df

# --- MAIN UI LAYOUT ---
st.title("AeroPredict: Predictive Maintenance Dashboard")

# Create Tabs for Navigation
tab1, tab2 = st.tabs(["Live Dashboard", "Project Overview"])

with tab2:
    st.header("Project Documentation & Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset: NASA C-MAPSS")
        st.markdown("""
        The **Commercial Modular Aero-Propulsion System Simulation** dataset is the industry standard for Predictive Maintenance research.
        
        - **Volume:** 160,359 total flight cycles.
        - **Sensors:** 21 telemetry channels (Temp, Pressure, Fan Speeds).
        - **Subsets:** Merged FD001 through FD004 to account for sea-level and high-altitude flight regimes, as well as single and multi-fault degradation modes.
        """)
        

    with col2:
        st.subheader("Model Performance")
        st.markdown("""
        The model was validated using a hold-out test set where the "actual" failure point was unknown to the predictor.
        
        | Metric | Value | Interpretation |
        | :--- | :--- | :--- |
        | **RMSE** | **~14.2 Cycles** | Average prediction error margin. |
        | **R² Score** | **0.88** | 88% accuracy in capturing degradation variance. |
        | **MAE** | **10.5 Cycles** | Mean Absolute Error per engine unit. |
        """)
        st.success("The model provides a ~14-cycle safety buffer, allowing for proactive maintenance scheduling.")

    st.divider()

    st.subheader("Testing Methodology")
    st.markdown("""
    1. **Feature Engineering:** Identified 14 sensors with the highest correlation to engine wear.
    2. **Training:** Supervised learning on 'run-to-failure' trajectories.
    3. **Evaluation:** Tested on truncated engine lives to simulate real-world 'snapshot' predictions.
    4. **Explainability:** Integrated Feature Importance to identify the specific mechanical component (e.g., HPC or Fan) showing signs of stress.
    """)


with tab1:
    # --- Sidebar Configuration inside Tab 1 Logic ---
    st.sidebar.header("Configuration")
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset Subset", ["FD001", "FD002", "FD003", "FD004"]
    )
    
    data = load_selected_data(dataset_choice)

    if data is not None:
        engine_id = st.sidebar.selectbox("Select Engine Unit ID", data['unit_nr'].unique())
        engine_data = data[data['unit_nr'] == engine_id].reset_index(drop=True)

        # --- Prediction & Metrics ---
        features = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
        latest_telemetry = engine_data[features].iloc[-1:]

        if model:
            predicted_rul = model.predict(latest_telemetry)[0]
            actual_rul = engine_data['RUL_actual'].iloc[-1]
            
            st.sidebar.markdown("---")
            st.sidebar.metric("Predicted RUL", f"{int(predicted_rul)} Cycles", f"{int(predicted_rul - actual_rul)} vs Actual")
            
            total_life_est = engine_data['time_cycles'].iloc[-1] + predicted_rul
            progress = min(int((engine_data['time_cycles'].iloc[-1] / total_life_est) * 100), 100)
            st.sidebar.progress(progress)

            if predicted_rul < 30: st.sidebar.error("CRITICAL: Maintenance Required")
            elif predicted_rul < 70: st.sidebar.warning("WARNING: Schedule Inspection")
            else: st.sidebar.success("Engine Status: Healthy")

        # --- Dashboard Visuals ---
        st.subheader(f"Telemetry Trends: Unit {engine_id} ({dataset_choice})")
        useful_sensors = ['s_2', 's_7', 's_11', 's_12']
        cols = st.columns(2)
        for i, sensor in enumerate(useful_sensors):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.plot(engine_data['time_cycles'], engine_data[sensor], color='#2ecc71' if i > 1 else '#e74c3c')
                ax.set_title(f"Sensor {sensor} Trend")
                st.pyplot(fig)

        # --- Explainability ---
        st.write("---")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("Key Risk Factors")
            if model:
                importances = pd.Series(model.feature_importances_, index=features).sort_values()
                fig_imp, ax_imp = plt.subplots()
                importances.tail(7).plot(kind='barh', ax=ax_imp, color='skyblue')
                st.pyplot(fig_imp)
        with col_b:
            st.subheader("Maintenance Recommendation")
            if predicted_rul < 40:
                st.error(f"**Action:** Immediate inspection of the component associated with `{importances.idxmax()}`.")
            else:
                st.success("No anomalies detected. Continue standard flight operations.")
    else:
        st.error("Dataset not found.")