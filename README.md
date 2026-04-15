# AeroPredict: Aircraft Engine Predictive Maintenance

AeroPredictis a Machine Learning-powered "Digital Twin" dashboard that predicts the **Remaining Useful Life (RUL)** of turbofan engines. Built with Python and Streamlit, it leverages the industry-standard **NASA C-MAPSS dataset** to provide real-time health monitoring and failure forecasting.

## Key Features
- **Multi-Dataset Support:** Trained on all 4 NASA subsets (FD001–FD004), handling various flight conditions (Sea Level to High Altitude) and multiple fault modes.
- **Explainable AI (XAI):** Uses Random Forest Feature Importance to identify exactly which sensors (e.g., HPC Outlet Pressure) are driving the degradation.
- **Dynamic Risk Assessment:** Real-time visual alerts (Healthy, Warning, Critical) based on predicted cycles.
- **Telemetry Trends:** Interactive visualization of critical sensor signals.

## Model Architecture
- **Algorithm:** Random Forest Regressor (150 estimators).
- **Training Data:** 160,000+ flight cycles from the C-MAPSS dataset.
- **Features:** 14 high-variance sensors identified via exploratory data analysis.

## Dataset: NASA C-MAPSS

The Commercial Modular Aero-Propulsion System Simulation dataset is the gold standard for mechanical failure prediction.

- **Scale:** Over 160,000 flight cycles of telemetry data.

- **Sensors:** 21 sensors capturing temperature, pressure, and fan speeds.

- **Complexity:** Merged four sub-datasets (FD001–FD004) to ensure the model handles both simple sea-level conditions and complex high-altitude operations with multiple simultaneous fault modes.

## Testing Methodology

We didn't just test on the data the model already saw. We used a Hold-Out Validation strategy:

**Training:** The model learned patterns from the *train_FD00x.txt* files, where engines run until they completely fail.

**Testing**: Evaluated the model using *test_FD00x.txt.* In these files, the data stops at a random point before failure.

**Ground Truth:** Compared the model's "guess" against the *RUL_FD00x.txt* files, which contain the actual remaining life for those test engines.

## Performance & Accuracy

For a regression task like this, the "Accuracy Score" is measured by how close the prediction is to the actual remaining cycles.

| Metric  | Score (Approx) | Meaning |
| ------------- | ------------- | 
| RMSE  | 14.2 Cycles  | On average, the model is off by about 14 flights. |
| R<sup>2</sup> Score  | 0.88  | 88% of the variance in engine degradation is captured by the model. |
| MAE  | 10.5 Cycles  | The typical absolute error per prediction. |

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/HussamDroid/AeroPredict.git](https://github.com/HussamDroid/AeroPredict.git)
   cd AeroPredict

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Train the Universal Model:**
   ```bash
   python train_model.py

4. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   