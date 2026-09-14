import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# ==========================================
# 1. LOAD TRAINING DATA
# ==========================================
data = pd.read_csv(
    "data/aws_training_data.csv"
)

# ==========================================
# 2. CREATE TIME-BASED FEATURES
# ==========================================
data["temp_change"] = data["temperature"].diff()
data["humidity_change"] = data["humidity"].diff()
data["pressure_change"] = data["pressure"].diff()

data["temp_rolling_std"] = (
    data["temperature"]
    .rolling(window=5)
    .std()
)

data["humidity_rolling_std"] = (
    data["humidity"]
    .rolling(window=5)
    .std()
)

# Missing sensor reading
data["is_missing"] = data[
    ["temperature", "humidity", "pressure"]
].isna().any(axis=1).astype(int)

# ==========================================
# 3. TRAIN ONLY ON NORMAL DATA
# ==========================================
normal_data = data[
    data["anomaly"] == 0
].copy()

# Fill missing values
normal_data = normal_data.fillna(0)

# ==========================================
# 4. SELECT FEATURES
# ==========================================
features = [
    "temperature",
    "humidity",
    "pressure",
    "temp_change",
    "humidity_change",
    "pressure_change",
    "temp_rolling_std",
    "humidity_rolling_std",
    "is_missing"
]

X = normal_data[features]

# ==========================================
# 5. TRAIN ISOLATION FOREST
# ==========================================
model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)

model.fit(X)

# ==========================================
# 6. SAVE MODEL
# ==========================================
joblib.dump(
    {
        "model": model,
        "features": features
    },
    "model/anomaly_model.pkl"
)

print("\n==========================================")
print("      SKYGUARD AI ANOMALY MODEL")
print("==========================================\n")

print("Normal training records:", len(normal_data))
print("Features used:", len(features))

print("\nAnomaly detection model trained successfully!")
print("Model saved to: model/anomaly_model.pkl")