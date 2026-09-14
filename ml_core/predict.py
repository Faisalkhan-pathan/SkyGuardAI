import pandas as pd
import joblib

from decision_engine import make_decision
from explanation_engine import generate_explanation
from neighbor_validator import validate_with_neighbors


# ==========================================
# 1. LOAD MODELS
# ==========================================

anomaly_data = joblib.load(
    "model/anomaly_model.pkl"
)

anomaly_model = anomaly_data["model"]
anomaly_features = anomaly_data["features"]


classifier_data = joblib.load(
    "model/classifier_model.pkl"
)

classifier = classifier_data["model"]
classifier_features = classifier_data["features"]


# ==========================================
# 2. ENTER 5 AWS READINGS
# ==========================================

print("\nEnter 5 consecutive AWS readings")
print("(Temperature, Humidity, Pressure)\n")

readings = []

for i in range(5):

    print(f"Reading {i + 1}")

    temperature = float(
        input("Temperature (°C): ")
    )

    humidity = float(
        input("Humidity (%): ")
    )

    pressure = float(
        input("Pressure (hPa): ")
    )

    readings.append([
        temperature,
        humidity,
        pressure
    ])

    print()


# ==========================================
# 3. CREATE DATAFRAME
# ==========================================

data = pd.DataFrame(
    readings,
    columns=[
        "temperature",
        "humidity",
        "pressure"
    ]
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
# 5. CHECK COMMUNICATION / INVALID DATA
# ==========================================

actual_missing = data[
    ["temperature", "humidity", "pressure"]
].isna().any(axis=1).any()


zero_packet = (
    (data["temperature"] == 0) &
    (data["humidity"] == 0) &
    (data["pressure"] == 0)
).any()


communication_error = (
    actual_missing or zero_packet
)


# ==========================================
# 6. DEFAULT RESULT
# ==========================================

anomaly_type = "Normal"
status = "NORMAL"
confidence = 100
severity = "LOW"


# ==========================================
# 7. COMMUNICATION ERROR
# ==========================================

if communication_error:

    anomaly_type = "Communication Error"
    status = "ANOMALY"
    confidence = 100
    severity = "HIGH"


# ==========================================
# 8. FROZEN SENSOR
# ==========================================

elif (
    data["temperature"].nunique() == 1
):

    anomaly_type = "Frozen Sensor"
    status = "ANOMALY"
    confidence = 100
    severity = "HIGH"


# ==========================================
# 9. TEMPERATURE SPIKE
# ==========================================

elif (
    data["temp_change"].abs() > 10
).any():

    anomaly_type = "Temperature Spike"
    status = "ANOMALY"
    confidence = 100
    severity = "HIGH"


# ==========================================
# 10. SENSOR DRIFT
# ==========================================

elif (
    (data["temp_change"] > 0.3).sum() >= 3
    or
    (data["temp_change"] < -0.3).sum() >= 3
):

    anomaly_type = "Sensor Drift"
    status = "ANOMALY"
    confidence = 100
    severity = "MEDIUM"


# ==========================================
# 11. MULTIVARIATE INCONSISTENCY
# ==========================================

elif (
    (data["pressure"] < 850).any()
    or
    (data["pressure"] > 1100).any()
    or
    (data["humidity"] < 0).any()
    or
    (data["humidity"] > 100).any()
):

    anomaly_type = "Multivariate Inconsistency"
    status = "ANOMALY"
    confidence = 100
    severity = "HIGH"


# ==========================================
# 12. ML ANOMALY DETECTION
# ==========================================

else:

    # Fill missing values only for ML processing

    ml_data = data.fillna(0)

    # Use latest observation

    latest_ml = ml_data.iloc[-1]

    input_data = pd.DataFrame([
        latest_ml
    ])

    # Isolation Forest

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

    else:

        anomaly_type = "Normal"
        status = "NORMAL"
        confidence = 100
        severity = "LOW"


# ==========================================
# 13. SEVERITY
# ==========================================

if anomaly_type == "Normal":

    severity = "LOW"

elif anomaly_type in [
    "Temperature Spike",
    "Communication Error",
    "Multivariate Inconsistency",
    "Frozen Sensor"
]:

    severity = "HIGH"

elif anomaly_type == "Sensor Drift":

    severity = "MEDIUM"

else:

    if confidence >= 80:
        severity = "HIGH"

    elif confidence >= 60:
        severity = "MEDIUM"

    else:
        severity = "LOW"


# ==========================================
# 14. GET LATEST READING
# ==========================================

latest = data.iloc[-1]


# ==========================================
# 15. NEIGHBOR STATION VALIDATION
# ==========================================

neighbor_result = None

# Only ask for neighbors when an anomaly exists

if status == "ANOMALY":

    print("\n==========================================")
    print("       NEIGHBOR STATION VALIDATION")
    print("==========================================")

    print(
        "\nEnter nearby AWS station readings."
    )

    print(
        "Enter 0 when you do not have neighbor data.\n"
    )

    neighbor_data = []

    for i in range(3):

        print(f"Neighbor Station {i + 1}")

        neighbor_temperature = float(
            input("Temperature (°C): ")
        )

        neighbor_humidity = float(
            input("Humidity (%): ")
        )

        neighbor_pressure = float(
            input("Pressure (hPa): ")
        )

        # No neighbor data option

        if (
            neighbor_temperature == 0
            and
            neighbor_humidity == 0
            and
            neighbor_pressure == 0
        ):
            break

        neighbor_data.append({
            "temperature":
                neighbor_temperature,

            "humidity":
                neighbor_humidity,

            "pressure":
                neighbor_pressure
        })

        print()


    # Run validator

    neighbor_result = validate_with_neighbors(

        station_temperature=latest["temperature"],

        station_humidity=latest["humidity"],

        station_pressure=latest["pressure"],

        neighbor_data=neighbor_data
    )


# ==========================================
# 16. CONTEXTUAL INTERPRETATION
# ==========================================

context = "Not Evaluated"


if status == "NORMAL":

    context = "Normal Weather Observation"


elif neighbor_result is not None:

    if neighbor_result["neighbor_status"] == "SUPPORTED":

        context = "Likely Genuine Weather Event"

    elif neighbor_result["neighbor_status"] == "NOT_SUPPORTED":

        context = "Likely Sensor Fault"

    else:

        context = "Uncertain"


# ==========================================
# 17. DECISION ENGINE
# ==========================================

decision = make_decision(

    anomaly_type=anomaly_type,

    confidence=confidence,

    temperature=latest["temperature"],

    humidity=latest["humidity"],

    pressure=latest["pressure"],

    context=context
)


# ==========================================
# 18. EXPLANATION ENGINE
# ==========================================

explanation = generate_explanation(

    anomaly_type=anomaly_type,

    confidence=confidence,

    temperature=latest["temperature"],

    humidity=latest["humidity"],

    pressure=latest["pressure"],

    data=data,

    context=context,

    neighbor_result=neighbor_result
)


# ==========================================
# 19. DISPLAY FINAL RESULT
# ==========================================

print("\n==========================================")
print("          SKYGUARD AI RESULT")
print("==========================================")

print(
    f"\nLatest Temperature : "
    f"{latest['temperature']:.2f} °C"
)

print(
    f"Latest Humidity    : "
    f"{latest['humidity']:.2f} %"
)

print(
    f"Latest Pressure    : "
    f"{latest['pressure']:.2f} hPa"
)

print(
    f"\nStatus      : {decision['status']}"
)

print(
    f"Type        : {anomaly_type}"
)

print(
    f"Category    : {decision['category']}"
)

print(
    f"Confidence  : {confidence:.2f}%"
)

print(
    f"Severity    : {decision['severity']}"
)

print(
    f"Context     : {context}"
)


# ==========================================
# 20. NEIGHBOR DETAILS
# ==========================================

if neighbor_result is not None:

    print("\n------------------------------------------")

    print("\nNEIGHBOR VALIDATION")

    print(
        f"Status : "
        f"{neighbor_result['neighbor_status']}"
    )

    if "neighbor_temperature_avg" in neighbor_result:

        print(
            f"Neighbor Avg Temperature : "
            f"{neighbor_result['neighbor_temperature_avg']:.2f} °C"
        )

        print(
            f"Temperature Difference   : "
            f"{neighbor_result['temperature_difference']:.2f} °C"
        )

        print(
            f"Neighbor Avg Humidity    : "
            f"{neighbor_result['neighbor_humidity_avg']:.2f} %"
        )

        print(
            f"Neighbor Avg Pressure    : "
            f"{neighbor_result['neighbor_pressure_avg']:.2f} hPa"
        )

    print(
        f"\n{neighbor_result['message']}"
    )


# ==========================================
# 21. EXPLANATION
# ==========================================

print("\n------------------------------------------")

print("\nWHY?")

print(
    explanation["why"]
)


print("\nRECOMMENDATION")

print(
    explanation["recommendation"]
)


print("\n==========================================")