import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib # To save the model

# 1. Load and Label Data
def prepare_data(file_path):
    col_names = ['unit_nr', 'time_cycles', 'set1', 'set2', 'set3'] + [f's_{i}' for i in range(1, 22)]
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    
    # Calculate RUL (Ground Truth)
    max_cycle = df.groupby('unit_nr')['time_cycles'].transform('max')
    df['RUL'] = max_cycle - df['time_cycles']
    return df

# 2. Feature Selection
# We use the sensors that showed clear trends in your UI
features = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

# Load data
train_data = prepare_data('CMAPSSData/train_FD001.txt')

X = train_data[features]
y = train_data['RUL']

# 3. Train the Model
print("Training the Random Forest model... 🛠️")
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

# 4. Save the model to a file
joblib.dump(model, 'rul_model.joblib')
print("Model saved as 'rul_model.joblib' ✅")