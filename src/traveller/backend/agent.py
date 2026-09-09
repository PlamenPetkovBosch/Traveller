"""
TravelAgent (Bulgaria, driving only)
-------------------------------------
Uses OpenRouteService (openrouteservice.org) for real driving distance,
duration, and route geometry between Bulgarian cities. ORS is free for
this volume of usage - no billing required, just a free API key.

Get a key at: https://openrouteservice.org/dev/#/signup
Then create a file named `.env` in the backend/ folder (see .env.example)
containing:

    ORS_API_KEY=your-key-here

and just run:

    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from the same folder as this file, regardless of the current
# working directory or how uvicorn's reloader spawns subprocesses.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

ORS_API_KEY = os.environ.get("ORS_API_KEY", "")
OTM_API_KEY = os.environ.get("OTM_API_KEY", "")
OTM_RADIUS_URL = "https://api.opentripmap.com/0.1/en/places/radius"

if ORS_API_KEY:
    print(f"[agent] Loaded ORS_API_KEY from {_ENV_PATH} (starts with: {ORS_API_KEY[:8]}...)")
else:
    print(f"[agent] WARNING: no ORS_API_KEY found. Looked for .env at: {_ENV_PATH}")

if OTM_API_KEY:
    print(f"[agent] Loaded OTM_API_KEY from {_ENV_PATH} (starts with: {OTM_API_KEY[:8]}...)")
else:
    print(f"[agent] WARNING: no OTM_API_KEY found. Top sights won't work until it's set.")
ORS_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"


@dataclass
class Destination:
    id: str
    name: str
    lat: float
    lon: float


# Major Bulgarian cities. Add more any time - just needs an id, name, and coordinates.
DESTINATIONS: list[Destination] = [
    Destination("sofia", "Sofia", 42.6977, 23.3219),
    Destination("plovdiv", "Plovdiv", 42.1354, 24.7453),
    Destination("varna", "Varna", 43.2141, 27.9147),
    Destination("burgas", "Burgas", 42.5048, 27.4626),
    Destination("ruse", "Ruse", 43.8564, 25.9704),
    Destination("stara_zagora", "Stara Zagora", 42.4258, 25.6345),
    Destination("pleven", "Pleven", 43.4170, 24.6067),
    Destination("sliven", "Sliven", 42.6818, 26.3294),
    Destination("dobrich", "Dobrich", 43.5726, 27.8273),
    Destination("shumen", "Shumen", 43.2712, 26.9364),
    Destination("pernik", "Pernik", 42.6053, 23.0378),
    Destination("veliko_tarnovo", "Veliko Tarnovo", 43.0757, 25.6172),
    Destination("blagoevgrad", "Blagoevgrad", 42.0195, 23.0943),
    Destination("vratsa", "Vratsa", 43.2094, 23.5544),
    Destination("gabrovo", "Gabrovo", 42.8747, 25.3188),
    Destination("yambol", "Yambol", 42.4840, 26.5036),
    Destination("kardzhali", "Kardzhali", 41.6500, 25.3778),
    Destination("vidin", "Vidin", 43.9910, 22.8763),
    Destination("montana", "Montana", 43.4125, 23.2249),
    Destination("silistra", "Silistra", 44.1167, 27.2622),
]

_BY_ID = {d.id: d for d in DESTINATIONS}


class TravelAgent:
    """Agent responsible for Bulgarian destination lookups and driving routes."""

    def list_destinations(self) -> list[Destination]:
        return DESTINATIONS

    def get_destination(self, dest_id: str) -> Destination | None:
        return _BY_ID.get(dest_id)

    def plan_trip(self, origin_id: str, destination_id: str) -> dict:
        """
        Given two destination ids, fetch a real driving route between them
        from OpenRouteService: distance, duration, and the road geometry
        so the frontend can draw the actual route (not a straight line).
        """
        origin = self.get_destination(origin_id)
        destination = self.get_destination(destination_id)

        if origin is None or destination is None:
            raise ValueError("Unknown destination id")

        if not ORS_API_KEY:
            raise RuntimeError(
                "No ORS_API_KEY set. Get a free key at "
                "https://openrouteservice.org/dev/#/signup and set it as an "
                "environment variable before starting the server."
            )

        # ORS expects [lon, lat] order, and start/end as separate coordinate pairs.
        body = {
            "coordinates": [
                [origin.lon, origin.lat],
                [destination.lon, destination.lat],
            ]
        }
        headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}

        resp = requests.post(ORS_DIRECTIONS_URL, json=body, headers=headers, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"Route request failed ({resp.status_code}): {resp.text[:300]}")

        data = resp.json()
        feature = data["features"][0]
        summary = feature["properties"]["summary"]
        # GeoJSON geometry is [lon, lat] pairs - flip to [lat, lon] for Leaflet.
        route_coords = [[lat, lon] for lon, lat in feature["geometry"]["coordinates"]]

        distance_km = summary["distance"] / 1000
        duration_minutes = summary["duration"] / 60

        return {
            "origin": _destination_to_dict(origin),
            "destination": _destination_to_dict(destination),
            "distance_km": round(distance_km, 1),
            "duration_minutes": round(duration_minutes),
            "route": route_coords,
        }

    def get_top_sights(self, destination_id: str, limit: int = 10) -> list[dict]:
        """
        Fetch notable attractions near a city from OpenTripMap, ranked by
        their importance rating, and return the top `limit` of them.
        """
        destination = self.get_destination(destination_id)
        if destination is None:
            raise ValueError("Unknown destination id")

        if not OTM_API_KEY:
            raise RuntimeError(
                "No OTM_API_KEY set. Get a free key at "
                "https://opentripmap.io/product and set it in your .env file "
                "as OTM_API_KEY=your-key-here."
            )

        params = {
            "radius": 10000,  # meters
            "lon": destination.lon,
            "lat": destination.lat,
            "kinds": "interesting_places",
            "rate": 2,  # only reasonably notable places
            "format": "json",
            "limit": 40,  # fetch extra, then rank locally and trim to `limit`
            "apikey": OTM_API_KEY,
        }
        resp = requests.get(OTM_RADIUS_URL, params=params, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"Sights request failed ({resp.status_code}): {resp.text[:300]}")

        places = resp.json()
        # Drop unnamed entries, then rank by importance rating (highest first).
        named = [p for p in places if p.get("name")]
        named.sort(key=lambda p: _parse_rate(p.get("rate")), reverse=True)

        return [
            {
                "xid": p["xid"],
                "name": p["name"],
                "kind": _primary_kind(p.get("kinds", "")),
                "lat": p["point"]["lat"],
                "lon": p["point"]["lon"],
            }
            for p in named[:limit]
        ]


def _parse_rate(rate) -> float:
    """OpenTripMap 'rate' can be an int, or strings like '1h', '2h', '3h', '7' (UNESCO).
    Higher is more notable. Unparseable values sort last."""
    if rate is None:
        return 0
    s = str(rate).rstrip("h")
    try:
        return float(s)
    except ValueError:
        return 0


def _primary_kind(kinds: str) -> str:
    """OpenTripMap returns a comma-separated tag list; show a readable first tag."""
    if not kinds:
        return "sight"
    first = kinds.split(",")[0]
    return first.replace("_", " ")


def _destination_to_dict(d: Destination) -> dict:
    return {"id": d.id, "name": d.name, "lat": d.lat, "lon": d.lon}
