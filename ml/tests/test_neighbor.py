from neighbor_validator import validate_with_neighbors


# ==========================================
# TEST 1: GENUINE WEATHER EVENT
# ==========================================

print("\n==========================================")
print(" TEST 1: GENUINE WEATHER EVENT")
print("==========================================")

result = validate_with_neighbors(
    station_temperature=45,
    station_humidity=60,
    station_pressure=1000,

    neighbor_data=[
        {
            "temperature": 44.5,
            "humidity": 61,
            "pressure": 1001
        },

        {
            "temperature": 45.2,
            "humidity": 59,
            "pressure": 1000
        },

        {
            "temperature": 44.8,
            "humidity": 60,
            "pressure": 1002
        }
    ]
)


for key, value in result.items():
    print(f"{key}: {value}")


# ==========================================
# TEST 2: POSSIBLE SENSOR FAULT
# ==========================================

print("\n==========================================")
print(" TEST 2: POSSIBLE SENSOR FAULT")
print("==========================================")

result = validate_with_neighbors(
    station_temperature=45,
    station_humidity=60,
    station_pressure=1000,

    neighbor_data=[
        {
            "temperature": 31,
            "humidity": 61,
            "pressure": 1001
        },

        {
            "temperature": 32,
            "humidity": 59,
            "pressure": 1000
        },

        {
            "temperature": 31.5,
            "humidity": 60,
            "pressure": 1002
        }
    ]
)


for key, value in result.items():
    print(f"{key}: {value}")


print("\n==========================================")