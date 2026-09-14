"""
SkyGuard AI — FastAPI Backend
SIH Hackathon Project

Integrates:
  - stations.StationManager   (station CRUD + neighbor lookup)
  - simulator.weather_simulator_loop  (real-time weather data feed)
  - SkyGuardAI.ml_engine.predict_aws  (anomaly detection + classification)
"""

import sys
import os
import asyncio

# ============================================================
# ML BRIDGE — resolve the nested SkyGuardAI path
# The ML teammate's code uses relative paths for model/ and data/,
# so we must chdir into that directory before importing.
# This mirrors the pattern proven in test_ml_import.py.
# ============================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

NESTED_ML_PATH = os.path.join(PROJECT_ROOT, "ml_core")

os.chdir(NESTED_ML_PATH)
sys.path.insert(0, NESTED_ML_PATH)

from ml_engine import predict_aws  # noqa: E402

# The backend directory must also be on sys.path so that stations.py
# and simulator.py remain importable after the chdir above.
sys.path.insert(0, BACKEND_DIR)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from stations import station_manager  # noqa: E402
import simulator  # noqa: E402

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SkyGuard AI",
    description="Automatic Weather Station anomaly detection system",
    version="1.0.0",
)

# ============================================================
# CORS — allow the frontend dashboard to connect
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STARTUP — launch the weather simulator in the background
# ============================================================

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulator.weather_simulator_loop())


# ============================================================
# ENDPOINTS
# ============================================================

# ---- Health Check ------------------------------------------

@app.get("/api/health")
async def health_check():
    """Basic liveness probe."""
    return {"status": "healthy"}


# ---- All Stations ------------------------------------------

@app.get("/api/stations")
async def get_stations():
    """Return the current state of every AWS station."""
    return station_manager.get_all_stations()


# ---- Single Station ----------------------------------------

@app.get("/api/stations/{station_id}")
async def get_station(station_id: str):
    """Return a single station by ID, or 404."""
    station = station_manager.get_station(station_id)
    if station is None:
        raise HTTPException(
            status_code=404,
            detail=f"Station {station_id} not found"
        )
    return station


# ---- ML Analysis -------------------------------------------

@app.get("/api/stations/{station_id}/analysis")
async def analyse_station(station_id: str):
    """
    Run the full SkyGuard AI ML pipeline on a station.

    1. Fetch the station's rolling history   → readings
    2. Fetch its nearest neighbors' readings → neighbor_data
    3. Pass both into predict_aws()
    4. If the ML returns ANOMALY, update the station status
    """
    station = station_manager.get_station(station_id)
    if station is None:
        raise HTTPException(
            status_code=404,
            detail=f"Station {station_id} not found"
        )

    target_history = station_manager.get_station_history(station_id)
    if not target_history:
        raise HTTPException(
            status_code=400,
            detail=f"No historical readings available for {station_id}"
        )

    neighbor_data = station_manager.get_nearby_readings(station_id)

    result = predict_aws(
        readings=target_history,
        neighbor_data=neighbor_data if neighbor_data else None,
    )

    # Sync the ML verdict back into the station manager
    if result.get("status") == "ANOMALY":
        station_manager.stations[station_id]["status"] = "ANOMALY"

    return result


# ---- Active Alerts -----------------------------------------

@app.get("/api/alerts")
async def get_alerts():
    """Return every station whose current status is ANOMALY."""
    return [
        station
        for station in station_manager.get_all_stations()
        if station["status"] == "ANOMALY"
    ]


# ---- Hidden: Force Anomaly for Demo -----------------------

@app.post("/api/test/force_anomaly/{station_id}")
async def force_anomaly(station_id: str):
    """
    Hidden endpoint for the judges' demo.
    Tells the simulator to inject an anomaly into the
    specified station on its next tick.
    """
    station = station_manager.get_station(station_id)
    if station is None:
        raise HTTPException(
            status_code=404,
            detail=f"Station {station_id} not found"
        )

    simulator.FORCE_ANOMALY_STATION = station_id

    return {
        "message": f"Anomaly injection armed for {station_id}",
        "station_id": station_id,
    }
