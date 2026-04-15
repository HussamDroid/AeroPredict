import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
import joblib

def load_all_datasets():
    all_data = []
    # Find all training files in the directory
    files = glob.glob('CMAPSSData/train_FD00*.txt')
    
    col_names = ['unit_nr', 'time_cycles', 'set1', 'set2', 'set3'] + [f's_{i}' for i in range(1, 22)]
    
    for file in files:
        df = pd.read_csv(file, sep='\s+', header=None, names=col_names)
        # Calculate RUL for each file individually before merging
        df['RUL'] = df.groupby('unit_nr')['time_cycles'].transform('max') - df['time_cycles']
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

# Features that exist across all datasets
features = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

print("Loading and merging all NASA C-MAPSS subsets... 📂")
full_train = load_all_datasets()

X = full_train[features]
y = full_train['RUL']

print(f"Training on {len(full_train)} total flight cycles... 🛠️")
# Increase n_estimators for the larger dataset
model = RandomForestRegressor(n_estimators=150, max_depth=12, n_jobs=-1, random_state=42)
model.fit(X, y)

joblib.dump(model, 'rul_model.joblib')
print("Universal Model saved successfully! ✅")