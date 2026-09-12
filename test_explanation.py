from explanation_engine import generate_explanation


result = generate_explanation(
    anomaly_type="Temperature Spike",
    confidence=100,
    temperature=58,
    humidity=65,
    pressure=1001
)


print("\n===================================")
print("       SKYGUARD EXPLANATION")
print("===================================")

print(f"Anomaly      : Temperature Spike")
print(f"Confidence   : 100%")

print(f"\nWhy?")
print(result["why"])

print(f"\nRecommended Action:")
print(result["recommendation"])

print("===================================")