import joblib
import pandas as pd

from decision_engine import build_decision
from explanation_engine import generate_explanation
from neighbor_validator import validate_with_neighbors


# ============================================================
# LOAD MODELS
# ============================================================

anomaly_data = joblib.load("model/anomaly_model.pkl")
anomaly_model = anomaly_data["model"]
anomaly_features = anomaly_data["features"]

classifier_data = joblib.load("model/classifier_model.pkl")
classifier = classifier_data["model"]
classifier_features = classifier_data["features"]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    data = data.copy()

    data["temp_change"] = (
        data["temperature"].diff().fillna(0)
    )

    data["humidity_change"] = (
        data["humidity"].diff().fillna(0)
    )

    data["pressure_change"] = (
        data["pressure"].diff().fillna(0)
    )

    data["temp_rolling_std"] = (
        data["temperature"]
        .rolling(window=3)
        .std()
        .fillna(0)
    )

    data["humidity_rolling_std"] = (
        data["humidity"]
        .rolling(window=3)
        .std()
        .fillna(0)
    )

    data["is_missing"] = (
        data[
            ["temperature", "humidity", "pressure"]
        ]
        .isna()
        .any(axis=1)
        .astype(int)
    )

    return data


# ============================================================
# MAIN ML FUNCTION
# ============================================================

def predict_aws(readings, neighbor_data=None):

    if not readings:
        raise ValueError(
            "At least one AWS reading is required."
        )

    data = pd.DataFrame(readings)

    required_columns = [
        "temperature",
        "humidity",
        "pressure"
    ]

    for column in required_columns:

        if column not in data.columns:

            raise ValueError(
                f"Missing required field: {column}"
            )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    data = create_features(data)

    latest = data.iloc[-1]

    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    status = "NORMAL"
    anomaly_type = "Normal"
    confidence = 100.0

    # ========================================================
    # RULE-BASED CHECKS
    # ========================================================

    # --------------------------------------------------------
    # 1. COMMUNICATION ERROR
    # --------------------------------------------------------

    if (
        data[
            ["temperature", "humidity", "pressure"]
        ]
        .isna()
        .any()
        .any()
    ):

        status = "ANOMALY"
        anomaly_type = "Communication Error"
        confidence = 100.0

    elif (
        (data["temperature"] == 0)
        &
        (data["humidity"] == 0)
        &
        (data["pressure"] == 0)
    ).any():

        status = "ANOMALY"
        anomaly_type = "Communication Error"
        confidence = 100.0

    # --------------------------------------------------------
    # 2. FROZEN SENSOR
    # --------------------------------------------------------

    elif data["temperature"].nunique() == 1:

        status = "ANOMALY"
        anomaly_type = "Frozen Sensor"
        confidence = 100.0

    # --------------------------------------------------------
    # 3. TEMPERATURE SPIKE
    # --------------------------------------------------------

    elif (
        data["temperature"]
        .diff()
        .abs()
        .max()
        > 10
    ):

        status = "ANOMALY"
        anomaly_type = "Temperature Spike"
        confidence = 100.0

    # --------------------------------------------------------
    # 4. SENSOR DRIFT
    # --------------------------------------------------------

    elif (
        data["temperature"]
        .diff()
        .abs()
        .gt(0.3)
        .sum()
        >= 3
    ):

        status = "ANOMALY"
        anomaly_type = "Sensor Drift"
        confidence = 100.0

    # --------------------------------------------------------
    # 5. MULTIVARIATE INCONSISTENCY
    # --------------------------------------------------------

    elif (
        (data["pressure"] < 850).any()
        or
        (data["pressure"] > 1100).any()
        or
        (data["humidity"] < 0).any()
        or
        (data["humidity"] > 100).any()
    ):

        status = "ANOMALY"
        anomaly_type = "Multivariate Inconsistency"
        confidence = 100.0

    # ========================================================
    # MACHINE LEARNING
    # ========================================================

    else:

        X_anomaly = data[anomaly_features]

        anomaly_prediction = anomaly_model.predict(
            X_anomaly
        )

        if -1 in anomaly_prediction:

            status = "ANOMALY"

            X_classifier = data[classifier_features]

            class_prediction = classifier.predict(
                X_classifier
            )

            probabilities = classifier.predict_proba(
                X_classifier
            )

            predicted_class = class_prediction[-1]

            confidence = (
                max(probabilities[-1]) * 100
            )

            anomaly_type = str(
                predicted_class
            )

            if anomaly_type == "Normal":

                anomaly_type = "Uncertain Anomaly"

        else:

            status = "NORMAL"
            anomaly_type = "Normal"
            confidence = 100.0

    # ========================================================
    # NEIGHBOR VALIDATION
    # ========================================================

    neighbor_result = None
    context = "Not Evaluated"

    if (
        status == "ANOMALY"
        and neighbor_data
    ):

        neighbor_result = validate_with_neighbors(

            station_temperature=float(
                latest["temperature"]
            ),

            station_humidity=float(
                latest["humidity"]
            ),

            station_pressure=float(
                latest["pressure"]
            ),

            neighbor_data=neighbor_data
        )

        if (
            neighbor_result["neighbor_status"]
            == "SUPPORTED"
        ):

            context = (
                "Likely Genuine Weather Event"
            )

        elif (
            neighbor_result["neighbor_status"]
            == "NOT_SUPPORTED"
        ):

            context = "Likely Sensor Fault"

        else:

            context = "Uncertain"

    # ========================================================
    # EXPLANATION ENGINE
    # ========================================================

    explanation = generate_explanation(

        anomaly_type=anomaly_type,

        confidence=confidence,

        temperature=float(
            latest["temperature"]
        ),

        humidity=float(
            latest["humidity"]
        ),

        pressure=float(
            latest["pressure"]
        ),

        data=data,

        context=context,

        neighbor_result=neighbor_result
    )

    # ========================================================
    # DECISION INFORMATION
    # ========================================================

    if status == "NORMAL":

        final_status = "NORMAL"
        final_severity = "LOW"

        reason = (
            "AWS observations follow a normal pattern."
        )

        recommendation = (
            "Continue normal monitoring."
        )

    elif context == "Likely Genuine Weather Event":

        final_status = "ANOMALY"
        final_severity = "HIGH"

        reason = (
            "The observation is unusual, but nearby "
            "AWS stations show similar conditions."
        )

        recommendation = (
            "Treat the observation as a possible "
            "genuine weather event and continue "
            "monitoring nearby AWS stations."
        )

    elif context == "Likely Sensor Fault":

        final_status = "ANOMALY"
        final_severity = "HIGH"

        reason = (
            "The observation is anomalous and nearby "
            "AWS stations do not show similar conditions."
        )

        recommendation = (
            "Inspect and calibrate the affected AWS sensor."
        )

    elif context == "Uncertain":

        final_status = "REVIEW"
        final_severity = "MEDIUM"

        reason = (
            "An unusual observation was detected, but "
            "there is insufficient contextual evidence "
            "to determine whether it is a genuine "
            "weather event or sensor fault."
        )

        recommendation = (
            "Review additional historical and nearby "
            "AWS observations."
        )

    else:

        final_status = status

        if anomaly_type == "Communication Error":

            final_severity = "HIGH"

            reason = (
                "Missing or zero-valued AWS data indicates "
                "a possible communication failure."
            )

            recommendation = (
                "Check the AWS communication link and "
                "data transmission system."
            )

        elif anomaly_type == "Frozen Sensor":

            final_severity = "HIGH"

            reason = (
                "Temperature remained unchanged across "
                "the recent observations."
            )

            recommendation = (
                "Inspect the temperature sensor for a "
                "frozen or malfunctioning state."
            )

        elif anomaly_type == "Temperature Spike":

            final_severity = "HIGH"

            reason = (
                "Temperature changed sharply within "
                "the recent observations."
            )

            recommendation = (
                "Inspect the temperature sensor and "
                "continue monitoring."
            )

        elif anomaly_type == "Sensor Drift":

            final_severity = "MEDIUM"

            reason = (
                "Temperature shows a consistent "
                "drifting pattern."
            )

            recommendation = (
                "Monitor the sensor and consider calibration."
            )

        elif anomaly_type == "Multivariate Inconsistency":

            final_severity = "HIGH"

            reason = (
                "One or more AWS observations contain "
                "values outside expected atmospheric ranges."
            )

            recommendation = (
                "Inspect the affected AWS observation "
                "and sensor/data-quality system."
            )

        else:

            final_severity = "MEDIUM"

            reason = (
                "An unusual AWS observation was detected."
            )

            recommendation = (
                "Review the observation and continue monitoring."
            )

    # ========================================================
    # DECISION ENGINE
    # ========================================================

    decision = build_decision(
        status=final_status,
        severity=final_severity,
        reason=reason,
        recommendation=recommendation,
        anomaly_type=anomaly_type,
        confidence=confidence
    )

    # ========================================================
    # FINAL CATEGORY
    # ========================================================

    if context == "Likely Genuine Weather Event":

        category = "Genuine Weather Event"

    elif context == "Likely Sensor Fault":

        category = "Sensor Fault"

    elif context == "Uncertain":

        category = "Uncertain"

    elif status == "NORMAL":

        category = "Normal"

    elif anomaly_type == "Communication Error":

        category = "Communication Fault"

    elif anomaly_type == "Multivariate Inconsistency":

        category = "Data Quality Issue"

    elif anomaly_type in [
        "Temperature Spike",
        "Frozen Sensor",
        "Sensor Drift"
    ]:

        category = "Sensor Fault"

    else:

        category = "Uncertain"

    # ========================================================
    # FINAL RESULT
    # ========================================================

    result = {

        "status": status,

        "anomaly_type": anomaly_type,

        "category": category,

        "confidence": round(
            float(confidence),
            2
        ),

        "severity": final_severity,

        "context": context,

        "explanation": explanation["why"],

        "recommendation": explanation[
            "recommendation"
        ]
    }

    # ========================================================
    # NEIGHBOR VALIDATION RESULT
    # ========================================================

    if neighbor_result is not None:

        result["neighbor_validation"] = {

            "status":
                neighbor_result[
                    "neighbor_status"
                ],

            "weather_support":
                neighbor_result[
                    "weather_support"
                ],

            "neighbor_temperature_avg":
                round(
                    neighbor_result[
                        "neighbor_temperature_avg"
                    ],
                    2
                ),

            "temperature_difference":
                round(
                    neighbor_result[
                        "temperature_difference"
                    ],
                    2
                ),

            "neighbor_humidity_avg":
                round(
                    neighbor_result[
                        "neighbor_humidity_avg"
                    ],
                    2
                ),

            "neighbor_pressure_avg":
                round(
                    neighbor_result[
                        "neighbor_pressure_avg"
                    ],
                    2
                )
        }

    return result