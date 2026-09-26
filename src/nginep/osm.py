"""OpenStreetMap lookups via the Overpass API.

Overpass rejects requests without a real User-Agent (the default curl/requests
UA gets a 406), so every request here sends one identifying the project.
"""
from __future__ import annotations

import time
from math import atan2, cos, radians, sin, sqrt

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "nginep-tanyainap-research/0.1 (+https://github.com/rafidhiyaulh/nginep)"

# Nominatim usage policy caps at 1 req/sec; this module makes few calls total
# (area centroids + occasional per-hotel fallback), so a simple module-level
# throttle is enough — no need for a job queue.
_last_nominatim_call = 0.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlmb / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


def geocode(query: str, bounded_to_bali: bool = True) -> dict | None:
    """Look up a place name via Nominatim. Rate-limited to ~1 req/sec."""
    global _last_nominatim_call
    elapsed = time.monotonic() - _last_nominatim_call
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)
    params = {"q": query, "format": "json", "limit": 1}
    if bounded_to_bali:
        # lon_min,lat_min,lon_max,lat_max covering all of Bali with margin
        params["viewbox"] = "114.3,-9.0,115.8,-8.0"
        params["bounded"] = 1
    resp = requests.get(NOMINATIM_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
    _last_nominatim_call = time.monotonic()
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None
    r = results[0]
    return {"lat": float(r["lat"]), "lon": float(r["lon"]), "display_name": r["display_name"]}


def fetch_hotels(area_name: str = "Bali", admin_level: str = "4", timeout: int = 60, retries: int = 3) -> list[dict]:
    """Fetch all tourism=hotel nodes/ways inside the named admin area.

    Returns a list of dicts with: osm_type, osm_id, name, lat, lon, tags.
    Ways are returned with their computed center point (via `out center`).
    The public Overpass instance is shared infra and occasionally times out
    under load (504), so this retries with backoff rather than failing the
    whole pipeline on a transient blip.
    """
    query = f"""
    [out:json][timeout:{timeout}];
    area["name"="{area_name}"]["admin_level"="{admin_level}"]->.a;
    (
      node["tourism"="hotel"](area.a);
      way["tourism"="hotel"](area.a);
    );
    out center tags;
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=timeout + 10,
            )
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < retries:
                wait = 5 * attempt
                print(f"  Overpass request failed ({e}); retrying in {wait}s ({attempt}/{retries})...")
                time.sleep(wait)
    else:
        raise last_error

    elements = resp.json().get("elements", [])

    results = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:  # way -> use computed center
            center = el.get("center", {})
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue
        results.append(
            {
                "osm_type": el["type"],
                "osm_id": el["id"],
                "name": name,
                "lat": lat,
                "lon": lon,
                "addr_city": tags.get("addr:city", ""),
                "addr_suburb": tags.get("addr:suburb", ""),
                "stars": tags.get("stars", ""),
            }
        )
    return results
