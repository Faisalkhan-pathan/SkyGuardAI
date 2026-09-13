import pandas as pd
import numpy as np


# ==========================================
# 1. SETTINGS
# ==========================================

np.random.seed(42)

n = 1000


# ==========================================
# 2. CREATE REALISTIC NORMAL AWS DATA
# ==========================================

timestamps = pd.date_range(
    start="2026-01-01",
    periods=n,
    freq="15min"
)

time = np.arange(n)


# Temperature follows a smooth daily pattern
temperature = (
    28
    + 6 * np.sin(2 * np.pi * time / 96)
    + np.random.normal(0, 0.4, n)
)


# Humidity generally moves opposite to temperature
humidity = (
    70
    - 15 * np.sin(2 * np.pi * time / 96)
    + np.random.normal(0, 1.5, n)
)


# Pressure changes slowly
pressure = (
    1000
    + 3 * np.sin(2 * np.pi * time / 384)
    + np.random.normal(0, 0.5, n)
)


# Keep values within realistic ranges
temperature = np.clip(
    temperature,
    15,
    45
)

humidity = np.clip(
    humidity,
    20,
    100
)

pressure = np.clip(
    pressure,
    980,
    1020
)


# ==========================================
# 3. CREATE DATAFRAME
# ==========================================

data = pd.DataFrame({
    "station_id": ["AWS001"] * n,
    "timestamp": timestamps,
    "temperature": temperature,
    "humidity": humidity,
    "pressure": pressure,
    "anomaly": [0] * n,
    "anomaly_type": ["Normal"] * n
})


# ==========================================
# 4. TEMPERATURE SPIKE
# ==========================================

spike_indices = [200, 201, 500, 501, 502]

for i in spike_indices:

    data.loc[i, "temperature"] = 58

    data.loc[i, "anomaly"] = 1

    data.loc[i, "anomaly_type"] = "Temperature Spike"


# ==========================================
# 5. FROZEN SENSOR
# ==========================================

frozen_indices = range(600, 620)

frozen_value = data.loc[599, "temperature"]

for i in frozen_indices:

    data.loc[i, "temperature"] = frozen_value

    data.loc[i, "anomaly"] = 1

    data.loc[i, "anomaly_type"] = "Frozen Sensor"


# ==========================================
# 6. SENSOR DRIFT
# ==========================================

drift_indices = range(700, 720)

starting_temperature = data.loc[699, "temperature"]

for position, i in enumerate(drift_indices):

    data.loc[i, "temperature"] = (
        starting_temperature
        + position * 0.5
    )

    data.loc[i, "anomaly"] = 1

    data.loc[i, "anomaly_type"] = "Sensor Drift"


# ==========================================
# 7. COMMUNICATION ERROR
# ==========================================

communication_indices = range(800, 810)

for i in communication_indices:

    data.loc[
        i,
        ["temperature", "humidity", "pressure"]
    ] = np.nan

    data.loc[i, "anomaly"] = 1

    data.loc[i, "anomaly_type"] = "Communication Error"


# ==========================================
# 8. MULTIVARIATE INCONSISTENCY
# ==========================================

inconsistent_indices = [900, 901, 902, 903]

for i in inconsistent_indices:

    data.loc[i, "temperature"] = 44

    data.loc[i, "humidity"] = 98

    data.loc[i, "pressure"] = 1018

    data.loc[i, "anomaly"] = 1

    data.loc[
        i,
        "anomaly_type"
    ] = "Multivariate Inconsistency"


# ==========================================
# 9. SAVE NORMAL DATA
# ==========================================

normal_data = data[
    data["anomaly"] == 0
].copy()

normal_data.to_csv(
    "data/aws_normal_data.csv",
    index=False
)


# ==========================================
# 10. SAVE TRAINING DATA
# ==========================================

data.to_csv(
    "data/aws_training_data.csv",
    index=False
)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("      SKYGUARD AI DATA GENERATOR")
print("==========================================\n")

print("Dataset created successfully!")

print("\nTotal records:", len(data))

print("\nAnomaly distribution:")

print(
    data["anomaly_type"].value_counts()
)

print("\nFiles created:")

print("data/aws_normal_data.csv")
print("data/aws_training_data.csv")