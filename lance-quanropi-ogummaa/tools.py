"""
core/tools.py
=============
Specialized geospatial tools for the Gummaa-Atlas agent.

Tools:
  - geocode()               — lat/lng → structured place name via Nominatim
  - reverse_geocode()       — lat/lng → address string
  - bbox_filter()           — filter a list of MemoryRecords by bounding box
  - build_map_overlay()     — construct GeoJSON FeatureCollection for frontend map
  - pmtiles_url()           — return a PMTiles vector tile URL for a given region
  - haversine_distance()    — great-circle distance between two coordinates (km)
  - classify_region()       — tag a coordinate with a broad historical region label
"""

from __future__ import annotations

import json
import logging
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GeoPlace:
    display_name: str
    lat: float
    lng: float
    country: str
    region: str | None = None
    raw: dict | None = None


@dataclass
class GeoJSONFeature:
    feature_type: str         # "Point" | "Polygon" | etc.
    coordinates: Any
    properties: dict


# ---------------------------------------------------------------------------
# Nominatim Geocoding (OSM — no API key required)
# ---------------------------------------------------------------------------

NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
NOMINATIM_HEADERS = {"User-Agent": "GummaaAtlas/1.0 (educational research)"}


def _nominatim_get(url: str) -> dict | list:
    req = urllib.request.Request(url, headers=NOMINATIM_HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def geocode(place_name: str) -> GeoPlace | None:
    """
    Convert a place name to a GeoPlace.

    Example::
        place = geocode("Kumasi, Ghana")
        # GeoPlace(display_name='Kumasi, Ashanti Region, Ghana', lat=6.688, lng=-1.624, ...)
    """
    encoded = urllib.parse.quote(place_name)
    url = (
        f"{NOMINATIM_BASE}/search"
        f"?q={encoded}&format=json&addressdetails=1&limit=1"
    )
    try:
        results = _nominatim_get(url)
    except Exception as exc:
        logger.warning("Geocode failed for '%s': %s", place_name, exc)
        return None

    if not results:
        return None

    r = results[0]
    addr = r.get("address", {})
    return GeoPlace(
        display_name=r.get("display_name", place_name),
        lat=float(r["lat"]),
        lng=float(r["lon"]),
        country=addr.get("country", ""),
        region=addr.get("state") or addr.get("region"),
        raw=r,
    )


def reverse_geocode(lat: float, lng: float) -> str:
    """
    Convert lat/lng to a human-readable address string.
    Returns a fallback string on error.
    """
    url = (
        f"{NOMINATIM_BASE}/reverse"
        f"?lat={lat}&lon={lng}&format=json&zoom=10"
    )
    try:
        r = _nominatim_get(url)
        return r.get("display_name", f"{lat:.4f}, {lng:.4f}")
    except Exception as exc:
        logger.warning("Reverse geocode failed: %s", exc)
        return f"{lat:.4f}, {lng:.4f}"


# ---------------------------------------------------------------------------
# Bounding Box Filter
# ---------------------------------------------------------------------------

def bbox_filter(
    records: list,
    min_lat: float,
    max_lat: float,
    min_lng: float,
    max_lng: float,
    lat_key: str = "lat",
    lng_key: str = "lng",
) -> list:
    """
    Filter a list of dicts (or objects with lat/lng attrs) to those falling
    within the given bounding box.

    Works with plain dicts or objects that store coordinates in their content
    as ``{"lat": ..., "lng": ...}`` JSON sub-strings.
    """
    filtered = []
    for record in records:
        # Try dict-style access first
        if isinstance(record, dict):
            lat = record.get(lat_key)
            lng = record.get(lng_key)
        else:
            lat = getattr(record, lat_key, None)
            lng = getattr(record, lng_key, None)

        if lat is None or lng is None:
            # Attempt to extract from content string
            content = getattr(record, "content", "") or record.get("content", "")
            try:
                payload = json.loads(content)
                lat = payload.get("lat")
                lng = payload.get("lng")
            except (json.JSONDecodeError, TypeError):
                pass

        if lat is None or lng is None:
            continue

        if min_lat <= float(lat) <= max_lat and min_lng <= float(lng) <= max_lng:
            filtered.append(record)

    return filtered


# ---------------------------------------------------------------------------
# GeoJSON Builder
# ---------------------------------------------------------------------------

def build_map_overlay(
    points: list[dict],
    collection_name: str = "Gummaa Results",
) -> dict:
    """
    Build a GeoJSON FeatureCollection from a list of point dicts.

    Each dict should contain at minimum: lat, lng, and optionally title/description.

    Returns a GeoJSON-spec dict ready to be sent to the frontend.
    """
    features = []
    for p in points:
        lat   = p.get("lat") or p.get("latitude")
        lng   = p.get("lng") or p.get("longitude")
        if lat is None or lng is None:
            continue
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lng), float(lat)],
            },
            "properties": {
                "title":       p.get("title", ""),
                "description": p.get("description", ""),
                "importance":  p.get("importance", 0.5),
                "scope":       p.get("scope", "historical"),
                "timestamp":   p.get("timestamp"),
            },
        }
        features.append(feature)

    return {
        "type":     "FeatureCollection",
        "name":     collection_name,
        "features": features,
    }


# ---------------------------------------------------------------------------
# PMTiles URL builder
# ---------------------------------------------------------------------------

PMTILES_BASE = "https://r2-public.protomaps.com/protomaps-sample-datasets"


def pmtiles_url(region: str = "africa") -> str:
    """
    Return a public PMTiles URL for the given broad region identifier.
    Clients can pass this to MapLibre-GL / Protomaps for vector tile rendering.
    """
    region_map = {
        "africa":      "protomaps2020-africa-extract.pmtiles",
        "west-africa": "protomaps2020-africa-extract.pmtiles",
        "world":       "nz-buildings-outlines.pmtiles",   # demo fallback
    }
    fname = region_map.get(region.lower(), region_map["world"])
    return f"{PMTILES_BASE}/{fname}"


# ---------------------------------------------------------------------------
# Haversine Distance
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Compute the great-circle distance between two (lat, lng) points in kilometres.
    """
    R = 6_371.0  # Earth radius km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Historical Region Classifier
# ---------------------------------------------------------------------------

# Rough bounding boxes for broad historical regions — extend as needed
_HISTORICAL_REGIONS: list[dict] = [
    {"name": "West Africa",       "min_lat": 4.0,  "max_lat": 23.0,  "min_lng": -17.5, "max_lng": 15.0},
    {"name": "East Africa",       "min_lat": -12.0,"max_lat": 22.0,  "min_lng": 29.0,  "max_lng": 51.0},
    {"name": "North Africa",      "min_lat": 20.0, "max_lat": 38.0,  "min_lng": -6.0,  "max_lng": 37.0},
    {"name": "Central Africa",    "min_lat": -10.0,"max_lat": 10.0,  "min_lng": 8.0,   "max_lng": 32.0},
    {"name": "Southern Africa",   "min_lat": -35.0,"max_lat": -1.0,  "min_lng": 10.0,  "max_lng": 40.0},
    {"name": "Maghreb",           "min_lat": 27.0, "max_lat": 38.0,  "min_lng": -6.0,  "max_lng": 13.0},
    {"name": "Horn of Africa",    "min_lat": -2.0, "max_lat": 18.0,  "min_lng": 38.0,  "max_lng": 52.0},
    {"name": "Sahel",             "min_lat": 10.0, "max_lat": 20.0,  "min_lng": -15.0, "max_lng": 40.0},
    {"name": "Nile Valley",       "min_lat": 10.0, "max_lat": 31.0,  "min_lng": 22.0,  "max_lng": 38.0},
    {"name": "Mediterranean",     "min_lat": 30.0, "max_lat": 47.0,  "min_lng": -6.0,  "max_lng": 37.0},
]


def classify_region(lat: float, lng: float) -> str:
    """
    Return the name of the broad historical region containing (lat, lng).
    Returns 'Unknown Region' if no match.
    """
    for reg in _HISTORICAL_REGIONS:
        if (reg["min_lat"] <= lat <= reg["max_lat"]
                and reg["min_lng"] <= lng <= reg["max_lng"]):
            return reg["name"]
    return "Unknown Region"
