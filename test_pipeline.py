import pandas as pd
import joblib


# ==========================================
# 1. LOAD BOTH MODELS
# ==========================================

anomaly_model = joblib.load(
    "model/anomaly_model.pkl"
)

classifier_data = joblib.load(
    "model/classifier_model.pkl"
)

classifier = classifier_data["model"]
features = classifier_data["features"]


# ==========================================
# 2. TEST SENSOR DATA
# ==========================================

test_data = pd.DataFrame({
    "temperature": [58.0],
    "humidity": [95.0],
    "pressure": [990.0]
})


# ==========================================
# 3. ANOMALY DETECTION
# ==========================================

anomaly_prediction = anomaly_model.predict(
    test_data
)[0]


# ==========================================
# 4. CREATE FEATURES FOR CLASSIFIER
# ==========================================

test_data["temp_change"] = 0
test_data["humidity_change"] = 0
test_data["pressure_change"] = 0
test_data["temp_rolling_std"] = 0
test_data["is_missing"] = 0


X = test_data[features]


# ==========================================
# 5. CLASSIFY ANOMALY
# ==========================================

if anomaly_prediction == -1:

    anomaly_type = classifier.predict(X)[0]

    probabilities = classifier.predict_proba(X)[0]

    confidence = max(probabilities) * 100

else:

    anomaly_type = "Normal"
    confidence = 100


# ==========================================
# 6. DETERMINE SEVERITY
# ==========================================

if anomaly_type == "Normal":
    severity = "LOW"

elif confidence >= 90:
    severity = "HIGH"

elif confidence >= 70:
    severity = "MEDIUM"

else:
    severity = "LOW"


# ==========================================
# 7. DISPLAY RESULT
# ==========================================

print("\n===================================")
print("       SKYGUARD AI RESULT")
print("===================================\n")

print(
    f"Temperature : {test_data.iloc[0]['temperature']} °C"
)

print(
    f"Humidity    : {test_data.iloc[0]['humidity']} %"
)

print(
    f"Pressure    : {test_data.iloc[0]['pressure']} hPa"
)

print("\n-----------------------------------")

if anomaly_prediction == -1:

    print("Status      : ANOMALY")
    print(f"Type        : {anomaly_type}")
    print(f"Confidence  : {confidence:.2f}%")
    print(f"Severity    : {severity}")

else:

    print("Status      : NORMAL")
    print("Confidence  : 100%")
    print("Severity    : LOW")

print("\n===================================")