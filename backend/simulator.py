"""
SkyGuard AI — Real-Time Weather Simulator

Generates a continuous stream of realistic sensor readings for every
AWS station using a mean-reverting random walk.  Each tick (4 s) nudges
temperature, humidity and pressure toward the ML training baselines
while adding bounded random noise.

The module-level FORCE_ANOMALY_STATION variable lets the hidden
/api/test/force_anomaly/{station_id} endpoint inject a sudden spike
into any station on the next tick.
"""

import asyncio
import random

from stations import station_manager

# ============================================================
# ANOMALY INJECTION HOOK
# Set to a station_id string to inject a spike on the next tick.
# main.py writes to this; the loop reads and resets it.
# ============================================================

FORCE_ANOMALY_STATION: str | None = None

# ============================================================
# ML-ALIGNED BASELINES  (from generate_ai.py)
# ============================================================

TEMP_CENTER = 28.0      # °C
TEMP_STEP = 0.4         # max random walk step
TEMP_MIN = 15.0
TEMP_MAX = 45.0

HUMIDITY_CENTER = 70.0   # %
HUMIDITY_STEP = 1.2
HUMIDITY_MIN = 20.0
HUMIDITY_MAX = 100.0

PRESSURE_CENTER = 1000.0  # hPa
PRESSURE_STEP = 0.4
PRESSURE_MIN = 980.0
PRESSURE_MAX = 1020.0

MEAN_REVERSION_STRENGTH = 0.05   # how hard values pull back to center

# ============================================================
# ANOMALY SPIKE VALUES  (matches test_ml_import.py pattern)
# ============================================================

SPIKE_TEMP = 42.0
SPIKE_HUMIDITY = 34.0
SPIKE_PRESSURE = 1002.0

# ============================================================
# TICK INTERVAL
# ============================================================

TICK_SECONDS = 4


# ============================================================
# HELPER — mean-reverting random step
# ============================================================

def _next_value(
    current: float,
    center: float,
    step: float,
    lo: float,
    hi: float,
) -> float:
    """
    Move *current* by a bounded random amount, with a gentle pull
    back toward *center* so the values don't wander too far from the
    ranges the ML models were trained on.
    """
    # Random walk component
    delta = random.uniform(-step, step)

    # Mean-reversion component — proportional to distance from center
    reversion = MEAN_REVERSION_STRENGTH * (center - current)

    new = current + delta + reversion

    # Hard clamp to physical limits
    return max(lo, min(hi, round(new, 2)))


# ============================================================
# MAIN LOOP
# ============================================================

async def weather_simulator_loop() -> None:
    """
    Runs forever as a background asyncio task.

    Every TICK_SECONDS:
      1. Iterate over every registered AWS station.
      2. Apply the mean-reverting random walk to each sensor channel.
      3. If FORCE_ANOMALY_STATION matches, inject the spike instead.
      4. Push the new reading into station_manager.
    """
    global FORCE_ANOMALY_STATION

    while True:
        await asyncio.sleep(TICK_SECONDS)

        for station in station_manager.get_all_stations():
            sid = station["station_id"]

            # --------------------------------------------------
            # Forced anomaly injection (hidden demo trigger)
            # --------------------------------------------------
            if FORCE_ANOMALY_STATION == sid:
                station_manager.update_station_reading(
                    station_id=sid,
                    temp=SPIKE_TEMP,
                    humidity=SPIKE_HUMIDITY,
                    pressure=SPIKE_PRESSURE,
                    status="ANOMALY",
                )
                FORCE_ANOMALY_STATION = None
                continue

            # --------------------------------------------------
            # Normal mean-reverting random walk
            # --------------------------------------------------
            new_temp = _next_value(
                station["temperature"],
                TEMP_CENTER,
                TEMP_STEP,
                TEMP_MIN,
                TEMP_MAX,
            )

            new_hum = _next_value(
                station["humidity"],
                HUMIDITY_CENTER,
                HUMIDITY_STEP,
                HUMIDITY_MIN,
                HUMIDITY_MAX,
            )

            new_press = _next_value(
                station["pressure"],
                PRESSURE_CENTER,
                PRESSURE_STEP,
                PRESSURE_MIN,
                PRESSURE_MAX,
            )

            station_manager.update_station_reading(
                station_id=sid,
                temp=new_temp,
                humidity=new_hum,
                pressure=new_press,
            )
