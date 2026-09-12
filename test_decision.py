from decision_engine import make_decision


result = make_decision(
    anomaly_type="Temperature Spike",
    confidence=100,
    temperature=58,
    humidity=65,
    pressure=1001
)

print("\n===================================")
print("       SKYGUARD DECISION")
print("===================================")

for key, value in result.items():

    print(f"{key}: {value}")

print("===================================")