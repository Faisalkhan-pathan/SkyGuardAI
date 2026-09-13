"""
SkyGuard AI - Decision and Explanation Engine

This module converts ML predictions and rule-based checks into
a structured final decision.

Architecture:

AWS Data
    ↓
Feature Engineering
    ↓
Isolation Forest
    ↓
Random Forest
    ↓
Decision Engine
    ↓
Explanation Engine
    ↓
Final Structured Result
"""

from typing import Optional


def build_decision(
    status: str,
    anomaly_type: str,
    confidence: float,
    severity: str,
    reason: str,
    recommendation: str,
) -> dict:
    """
    Create a standard SkyGuard AI decision.

    Keeping the output format centralized is important because
    the same structure will later be used by FastAPI and the
    frontend dashboard.
    """

    # Keep confidence safely between 0 and 1.
    confidence = max(0.0, min(1.0, float(confidence)))

    return {
        "status": status,
        "type": anomaly_type,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "severity": severity,
        "reason": reason,
        "recommendation": recommendation,
    }


def explain_result(
    anomaly_type: str,
    confidence: float,
    severity: Optional[str] = None,
) -> dict:
    """
    Generate a human-readable explanation and recommendation
    for a detected anomaly.
    """

    explanations = {
        "Normal": {
            "reason": "AWS observations appear consistent with expected sensor behavior.",
            "recommendation": "No immediate action required. Continue monitoring.",
            "severity": "LOW",
        },

        "Temperature Spike": {
            "reason": "Temperature changed by an unusually large amount between observations.",
            "recommendation": "Inspect the temperature sensor and verify the observation against nearby stations.",
            "severity": "HIGH",
        },

        "Frozen Sensor": {
            "reason": "Temperature remained constant across consecutive observations while other values changed.",
            "recommendation": "Inspect the temperature sensor for a stuck or frozen reading.",
            "severity": "HIGH",
        },

        "Sensor Drift": {
            "reason": "Temperature shows a sustained upward or downward change that may indicate sensor drift.",
            "recommendation": "Check sensor calibration and compare observations with nearby stations.",
            "severity": "MEDIUM",
        },

        "Communication Error": {
            "reason": "The observation packet contains missing or invalid communication values.",
            "recommendation": "Check the data logger and communication link before inspecting the physical sensor.",
            "severity": "HIGH",
        },

        "Multivariate Inconsistency": {
            "reason": "The combination of AWS observations contains values outside the expected physical range.",
            "recommendation": "Verify the pressure sensor and compare all observations with nearby stations.",
            "severity": "HIGH",
        },

        "Uncertain Anomaly": {
            "reason": "The observation appears unusual, but the available evidence is insufficient to confidently identify a specific fault.",
            "recommendation": "Review recent station history and compare with nearby stations before taking maintenance action.",
            "severity": "MEDIUM",
        },
    }

    explanation = explanations.get(
        anomaly_type,
        {
            "reason": "An unusual AWS observation was detected.",
            "recommendation": "Review the station observation and recent history.",
            "severity": "MEDIUM",
        },
    )

    # Allow the decision engine to override the default severity.
    final_severity = severity if severity is not None else explanation["severity"]

    return {
        "reason": explanation["reason"],
        "recommendation": explanation["recommendation"],
        "severity": final_severity,
    }


def create_final_decision(
    anomaly_type: str,
    confidence: float,
    status: Optional[str] = None,
    severity: Optional[str] = None,
) -> dict:
    """
    Convert an anomaly classification into the final SkyGuard AI
    decision format.

    This function is intentionally independent from the ML models.
    That means the ML code can remain unchanged while this layer
    evolves.
    """

    # Normal is handled explicitly.
    if anomaly_type == "Normal":
        final_status = "NORMAL"
    else:
        final_status = status if status is not None else "ANOMALY"

    explanation = explain_result(
        anomaly_type=anomaly_type,
        confidence=confidence,
        severity=severity,
    )

    return build_decision(
        status=final_status,
        anomaly_type=anomaly_type,
        confidence=confidence,
        severity=explanation["severity"],
        reason=explanation["reason"],
        recommendation=explanation["recommendation"],
    )


if __name__ == "__main__":
    """
    Simple standalone test.

    This does NOT run the ML models.
    It only checks whether the Decision + Explanation Engine
    itself is working correctly.
    """

    test_cases = [
        ("Normal", 1.00),
        ("Temperature Spike", 1.00),
        ("Frozen Sensor", 1.00),
        ("Sensor Drift", 1.00),
        ("Communication Error", 1.00),
        ("Multivariate Inconsistency", 0.95),
        ("Uncertain Anomaly", 0.70),
    ]

    print("\n" + "=" * 60)
    print("SKYGUARD AI - DECISION ENGINE TEST")
    print("=" * 60)

    for anomaly_type, confidence in test_cases:

        result = create_final_decision(
            anomaly_type=anomaly_type,
            confidence=confidence,
        )

        print(f"\nType           : {result['type']}")
        print(f"Status         : {result['status']}")
        print(f"Confidence     : {result['confidence_percent']:.2f}%")
        print(f"Severity       : {result['severity']}")
        print(f"Reason         : {result['reason']}")
        print(f"Recommendation : {result['recommendation']}")

    print("\n" + "=" * 60)
    print("DECISION ENGINE TEST COMPLETE")
    print("=" * 60)