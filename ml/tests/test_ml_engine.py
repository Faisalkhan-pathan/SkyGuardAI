from ml.ml_engine import predict_aws


readings = [
    {
        "temperature": 23,
        "humidity": 43,
        "pressure": 1000
    },
    {
        "temperature": 24,
        "humidity": 42,
        "pressure": 1001
    },
    {
        "temperature": 67,
        "humidity": 43,
        "pressure": 1003
    },
    {
        "temperature": 43,
        "humidity": 42,
        "pressure": 1002
    },
    {
        "temperature": 42,
        "humidity": 41,
        "pressure": 1005
    }
]


neighbor_data = [
    {
        "temperature": 45,
        "humidity": 34,
        "pressure": 1000
    },
    {
        "temperature": 43,
        "humidity": 34.8,
        "pressure": 1002.3
    },
    {
        "temperature": 43,
        "humidity": 34.7,
        "pressure": 1002.6
    }
]


result = predict_aws(
    readings,
    neighbor_data
)


print("\n==============================")
print("SKYGUARD AI ML ENGINE TEST")
print("==============================")

for key, value in result.items():
    print(f"{key}: {value}")