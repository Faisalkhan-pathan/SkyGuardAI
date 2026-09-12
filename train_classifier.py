import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# ==========================================
# 1. LOAD TRAINING DATA
# ==========================================

data = pd.read_csv("data/aws_training_data.csv")


# ==========================================
# 2. CREATE TIME-BASED FEATURES
# ==========================================

# Change in sensor values from previous reading
data["temp_change"] = data["temperature"].diff()
data["humidity_change"] = data["humidity"].diff()
data["pressure_change"] = data["pressure"].diff()


# Rolling temperature variation
data["temp_rolling_std"] = (
    data["temperature"]
    .rolling(window=5)
    .std()
)


# Detect missing sensor data
data["is_missing"] = data[
    ["temperature", "humidity", "pressure"]
].isna().any(axis=1).astype(int)


# Replace missing values
data = data.fillna(0)


# ==========================================
# 3. SELECT FEATURES
# ==========================================

features = [
    "temperature",
    "humidity",
    "pressure",
    "temp_change",
    "humidity_change",
    "pressure_change",
    "temp_rolling_std",
    "is_missing"
]

X = data[features]

y = data["anomaly_type"]


# ==========================================
# 4. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. TRAIN RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# ==========================================
# 6. TEST MODEL
# ==========================================

predictions = model.predict(X_test)

print("\n===== SKYGUARD AI CLASSIFIER =====\n")

print(classification_report(y_test, predictions))


# ==========================================
# 7. SAVE MODEL
# ==========================================

joblib.dump(
    {
        "model": model,
        "features": features
    },
    "model/classifier_model.pkl"
)

print("\nClassifier trained successfully!")
print("Model saved to: model/classifier_model.pkl")