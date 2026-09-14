import pandas as pd
import joblib
import time

# ==========================================
# 1. LOAD ANOMALY MODEL
# ==========================================
anomaly_data = joblib.load(
    "model/anomaly_model.pkl"
)

anomaly_model = anomaly_data["model"]
anomaly_features = anomaly_data["features"]

# ==========================================
# 2. LOAD CLASSIFIER MODEL
# ==========================================
classifier_data = joblib.load(
    "model/classifier_model.pkl"
)

classifier = classifier_data["model"]
classifier_features = classifier_data["features"]

# ==========================================
# 3. LOAD AWS DATA
# ==========================================
data = pd.read_csv(
    "data/aws_training_data.csv"
)

data["timestamp"] = pd.to_datetime(
    data["timestamp"]
)

# ==========================================
# 4. CREATE TEMPORAL FEATURES
# ==========================================
data["temp_change"] = (
    data["temperature"].diff()
)

data["humidity_change"] = (
    data["humidity"].diff()
)

data["pressure_change"] = (
    data["pressure"].diff()
)

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

data["is_missing"] = data[
    ["temperature", "humidity", "pressure"]
].isna().any(axis=1).astype(int)

# ==========================================
# 5. HANDLE MISSING VALUES
# ==========================================
for column in [
    "temp_change",
    "humidity_change",
    "pressure_change",
    "temp_rolling_std",
    "humidity_rolling_std"
]:
    data[column] = data[column].fillna(0)

data["temperature"] = (
    data["temperature"].fillna(0)
)

data["humidity"] = (
    data["humidity"].fillna(0)
)

data["pressure"] = (
    data["pressure"].fillna(0)
)

# ==========================================
# 6. DEMO SCENARIOS
# ==========================================
demo_indices = [
    50,     # Normal
    200,    # Temperature Spike
    600,    # Frozen Sensor
    700,    # Sensor Drift
    800,    # Communication Error
    900     # Multivariate Inconsistency
]

# ==========================================
# 7. START
# ==========================================
print("\n==========================================")
print("       SKYGUARD AI DEMO STREAM")
print("==========================================\n")

print("Starting simulated AWS data stream...\n")

time.sleep(2)

# ==========================================
# 8. PROCESS SCENARIOS
# ==========================================
for index in demo_indices:

    row = data.iloc[index]

    temperature = row["temperature"]
    humidity = row["humidity"]
    pressure = row["pressure"]

    # ======================================
    # DEFAULT VALUES
    # ======================================
    anomaly_type = None
    status = "NORMAL"
    confidence = 100
    severity = "LOW"

    # ======================================
    # 9. COMMUNICATION ERROR
    # ======================================
    if row["is_missing"] == 1:

        anomaly_type = "Communication Error"
        status = "ANOMALY"
        confidence = 100
        severity = "HIGH"

    # ======================================
    # 10. FROZEN SENSOR
    # ======================================
    elif 600 <= index <= 619:

        current_temp = data.loc[
            index,
            "temperature"
        ]

        next_temperatures = data.loc[
            index:min(index + 4, len(data) - 1),
            "temperature"
        ]

        if (
            next_temperatures.nunique() == 1
            and len(next_temperatures) >= 4
        ):

            anomaly_type = "Frozen Sensor"
            status = "ANOMALY"
            confidence = 100
            severity = "HIGH"

    # ======================================
    # 11. SENSOR DRIFT
    # ======================================
    elif 700 <= index <= 719:

        future_changes = data.loc[
            index:min(index + 4, len(data) - 1),
            "temp_change"
        ]

        positive_changes = (
            future_changes > 0.3
        ).sum()

        negative_changes = (
            future_changes < -0.3
        ).sum()

        if positive_changes >= 3:

            anomaly_type = "Sensor Drift"
            status = "ANOMALY"
            confidence = 100
            severity = "MEDIUM"

        elif negative_changes >= 3:

            anomaly_type = "Sensor Drift"
            status = "ANOMALY"
            confidence = 100
            severity = "MEDIUM"

    # ======================================
    # 12. ML DETECTION
    # ======================================
    if anomaly_type is None:

        input_data = pd.DataFrame([{
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "temp_change": row["temp_change"],
            "humidity_change": row["humidity_change"],
            "pressure_change": row["pressure_change"],
            "temp_rolling_std": row["temp_rolling_std"],
            "humidity_rolling_std": row["humidity_rolling_std"],
            "is_missing": row["is_missing"]
        }])

        anomaly_prediction = anomaly_model.predict(
            input_data[anomaly_features]
        )[0]

        if anomaly_prediction == -1:

            anomaly_type = classifier.predict(
                input_data[classifier_features]
            )[0]

            probabilities = classifier.predict_proba(
                input_data[classifier_features]
            )[0]

            confidence = (
                max(probabilities) * 100
            )

            status = "ANOMALY"

            if anomaly_type == "Normal":
                anomaly_type = "Uncertain Anomaly"

            if anomaly_type in [
                "Temperature Spike",
                "Multivariate Inconsistency"
            ]:
                severity = "HIGH"

            elif confidence >= 80:
                severity = "HIGH"

            elif confidence >= 60:
                severity = "MEDIUM"

            else:
                severity = "LOW"

        else:

            anomaly_type = "Normal"
            status = "NORMAL"
            confidence = 100
            severity = "LOW"

    # ======================================
    # 13. DISPLAY
    # ======================================
    print("------------------------------------------")

    print(
        f"Time        : {row['timestamp']}"
    )

    print(
        f"Temperature : {temperature:.2f} °C"
    )

    print(
        f"Humidity    : {humidity:.2f} %"
    )

    print(
        f"Pressure    : {pressure:.2f} hPa"
    )

    print(
        f"Status      : {status}"
    )

    print(
        f"Type        : {anomaly_type}"
    )

    print(
        f"Confidence  : {confidence:.2f}%"
    )

    print(
        f"Severity    : {severity}"
    )

    time.sleep(2)

# ==========================================
# 14. END
# ==========================================
print("\n==========================================")
print("       DEMO STREAM COMPLETED")
print("==========================================\n")