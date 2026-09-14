import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/anomaly_model.pkl")

# Different test scenarios
test_data = pd.DataFrame({
    "temperature": [
        31.5,   # Normal
        58.0,   # Temperature spike
        31.5,   # Frozen sensor
        45.0,   # Drift / unusual
        25.0,   # Normal
        44.0    # Multivariate inconsistency
    ],

    "humidity": [
        70,
        95,
        70,
        70,
        65,
        98
    ],

    "pressure": [
        998,
        990,
        998,
        995,
        1000,
        1018
    ]
})

# Run predictions
predictions = model.predict(test_data)

# Display results
print("\n===== SKYGUARD AI TEST =====\n")

for i, prediction in enumerate(predictions):

    if prediction == -1:
        status = "ANOMALY"
    else:
        status = "NORMAL"

    print(
        f"Test {i + 1}: "
        f"Temp={test_data.iloc[i]['temperature']}°C | "
        f"Humidity={test_data.iloc[i]['humidity']}% | "
        f"Pressure={test_data.iloc[i]['pressure']} hPa "
        f"→ {status}"
    )