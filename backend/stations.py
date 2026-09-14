import math
from datetime import datetime, timezone
from collections import deque

# Earth radius in kilometers for Haversine calculation
EARTH_RADIUS_KM = 6371.0

# 15 Regional Automatic Weather Stations (Warangal & Telangana Cluster)
INITIAL_STATIONS = [
    {"station_id": "AWS001", "station_name": "Warangal AWS", "district": "WARANGAL", "latitude": 17.9784, "longitude": 79.5941},
    {"station_id": "AWS002", "station_name": "Hanamkonda AWS", "district": "HANAMKONDA", "latitude": 17.9990, "longitude": 79.5800},
    {"station_id": "AWS003", "station_name": "Kazipet AWS", "district": "HANAMKONDA", "latitude": 17.9810, "longitude": 79.5280},
    {"station_id": "AWS004", "station_name": "Mulugu AWS", "district": "MULUGU", "latitude": 18.1920, "longitude": 79.9380},
    {"station_id": "AWS005", "station_name": "Jangaon AWS", "district": "JANGAON", "latitude": 17.7230, "longitude": 79.1620},
    {"station_id": "AWS006", "station_name": "Mahabubabad AWS", "district": "MAHABUBABAD", "latitude": 17.5980, "longitude": 80.0030},
    {"station_id": "AWS007", "station_name": "Narsampet AWS", "district": "WARANGAL", "latitude": 17.9250, "longitude": 79.8970},
    {"station_id": "AWS008", "station_name": "Parkal AWS", "district": "HANAMKONDA", "latitude": 18.2010, "longitude": 79.7120},
    {"station_id": "AWS009", "station_name": "Karimnagar AWS", "district": "KARIMNAGAR", "latitude": 18.4386, "longitude": 79.1288},
    {"station_id": "AWS010", "station_name": "Siddipet AWS", "district": "SIDDIPET", "latitude": 18.1018, "longitude": 78.8520},
    {"station_id": "AWS011", "station_name": "Khammam AWS", "district": "KHAMMAM", "latitude": 17.2473, "longitude": 80.1514},
    {"station_id": "AWS012", "station_name": "Bhupalpally AWS", "district": "JAYASHANKAR", "latitude": 18.4350, "longitude": 79.8660},
    {"station_id": "AWS013", "station_name": "Thorrur AWS", "district": "MAHABUBABAD", "latitude": 17.5680, "longitude": 79.6730},
    {"station_id": "AWS014", "station_name": "Station Ghanpur AWS", "district": "JANGAON", "latitude": 17.8420, "longitude": 79.3390},
    {"station_id": "AWS015", "station_name": "Huzurabad AWS", "district": "KARIMNAGAR", "latitude": 18.1900, "longitude": 79.3900},
]

# Baseline weather values derived from ML training dataset
DEFAULT_TEMP = 28.0
DEFAULT_HUMIDITY = 70.0
DEFAULT_PRESSURE = 1000.0

class StationManager:
    def __init__(self):
        self.stations = {}
        self.history = {}  # Rolling window buffer for ML feature engineering

        for s in INITIAL_STATIONS:
            sid = s["station_id"]
            self.stations[sid] = {
                **s,
                "temperature": DEFAULT_TEMP,
                "humidity": DEFAULT_HUMIDITY,
                "pressure": DEFAULT_PRESSURE,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "NORMAL"
            }
            # Maintain 5-10 historical readings for rolling feature calculations
            self.history[sid] = deque(maxlen=10)
            for _ in range(5):
                self.history[sid].append({
                    "temperature": DEFAULT_TEMP,
                    "humidity": DEFAULT_HUMIDITY,
                    "pressure": DEFAULT_PRESSURE
                })

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two geographic coordinates in kilometers."""
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return EARTH_RADIUS_KM * c

    def get_all_stations(self) -> list[dict]:
        return list(self.stations.values())

    def get_station(self, station_id: str) -> dict | None:
        return self.stations.get(station_id)

    def update_station_reading(self, station_id: str, temp: float, humidity: float, pressure: float, status: str = "NORMAL"):
        if station_id not in self.stations:
            return
        
        reading = {
            "temperature": round(temp, 2),
            "humidity": round(humidity, 2),
            "pressure": round(pressure, 2)
        }
        self.history[station_id].append(reading)
        
        self.stations[station_id]["temperature"] = reading["temperature"]
        self.stations[station_id]["humidity"] = reading["humidity"]
        self.stations[station_id]["pressure"] = reading["pressure"]
        self.stations[station_id]["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.stations[station_id]["status"] = status

    def get_station_history(self, station_id: str) -> list[dict]:
        return list(self.history.get(station_id, []))

    def get_nearby_readings(self, target_station_id: str, count: int = 3) -> list[dict]:
        """Find the nearest neighboring stations and return their latest telemetry."""
        target = self.stations.get(target_station_id)
        if not target:
            return []

        distances = []
        for sid, s in self.stations.items():
            if sid == target_station_id:
                continue
            dist = self.haversine_distance(target["latitude"], target["longitude"], s["latitude"], s["longitude"])
            distances.append((dist, sid))

        distances.sort(key=lambda x: x[0])
        nearest_ids = [sid for _, sid in distances[:count]]

        return [
            {
                "temperature": self.stations[nid]["temperature"],
                "humidity": self.stations[nid]["humidity"],
                "pressure": self.stations[nid]["pressure"]
            }
            for nid in nearest_ids
        ]

# Global singleton instance
station_manager = StationManager()
