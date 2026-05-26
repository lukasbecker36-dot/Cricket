"""Weather lookup for cricket venues using Open-Meteo (free, no API key).

Two endpoints used:
  - Geocoding: https://geocoding-api.open-meteo.com/v1/search
  - Historical archive: https://archive-api.open-meteo.com/v1/archive
  - Forecast: https://api.open-meteo.com/v1/forecast

Cached venue coordinates in models/venue_coords.json so we don't re-geocode.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request

logger = logging.getLogger(__name__)

VENUE_COORDS_FILE = Path("models/venue_coords.json")

# Hand-curated overrides for venues that geocoding doesn't resolve correctly
VENUE_OVERRIDES = {
    "M Chinnaswamy Stadium": (12.9788, 77.5995),  # Bengaluru
    "Wankhede Stadium": (18.9389, 72.8258),  # Mumbai
    "Eden Gardens": (22.5645, 88.3433),  # Kolkata
    "MA Chidambaram Stadium": (13.0631, 80.2789),  # Chennai
    "Narendra Modi Stadium": (23.0918, 72.5972),  # Ahmedabad
    "Arun Jaitley Stadium": (28.6378, 77.2436),  # Delhi
    "Brabourne Stadium": (18.9325, 72.8237),  # Mumbai
    "Rajiv Gandhi International Stadium": (17.4060, 78.5505),  # Hyderabad
    "Punjab Cricket Association Stadium": (30.6918, 76.7368),  # Mohali
    "Sawai Mansingh Stadium": (26.8943, 75.8061),  # Jaipur
    "Sheikh Zayed Stadium": (24.4248, 54.7227),  # Abu Dhabi
    "Dubai International Cricket Stadium": (25.0488, 55.2192),  # Dubai
    "Sharjah Cricket Stadium": (25.3409, 55.4275),  # Sharjah
    "Melbourne Cricket Ground": (-37.8200, 144.9834),
    "Sydney Cricket Ground": (-33.8915, 151.2247),
    "Adelaide Oval": (-34.9156, 138.5961),
    "Brisbane Cricket Ground": (-27.4858, 153.0381),  # The Gabba
    "Bellerive Oval": (-42.8772, 147.3737),  # Hobart
    "Manuka Oval": (-35.3175, 149.1347),  # Canberra
    "Docklands Stadium": (-37.8166, 144.9475),  # Marvel Stadium Melbourne
    "Perth Stadium": (-31.9509, 115.8893),
    "Optus Stadium": (-31.9509, 115.8893),
    "Lord's": (51.5293, -0.1727),
    "Kennington Oval": (51.4836, -0.1145),
    "Old Trafford": (53.4570, -2.2867),  # Manchester
    "Headingley": (53.8175, -1.5811),  # Leeds
    "Edgbaston": (52.4558, -1.9028),  # Birmingham
    "Trent Bridge": (52.9358, -1.1320),  # Nottingham
    "Sophia Gardens": (51.4811, -3.2014),  # Cardiff
    "Riverside Ground": (54.7717, -1.5380),  # Chester-le-Street
    "The Rose Bowl": (50.9259, -1.3197),  # Southampton
    "Ageas Bowl": (50.9259, -1.3197),
    "Grace Road": (52.6244, -1.1336),  # Leicester
    "County Ground, Bristol": (51.4778, -2.6175),
    "County Ground, Hove": (50.8295, -0.1601),
    "County Ground, Chelmsford": (51.7411, 0.4767),
    "County Ground, Northampton": (52.2374, -0.8909),
    "County Ground, Derby": (52.9203, -1.4670),
    "County Ground, New Road": (52.1898, -2.2273),  # Worcester
    "The Cooper Associates County Ground, Taunton": (51.0143, -3.1027),
    "Gaddafi Stadium": (31.5151, 74.3329),  # Lahore
    "National Stadium": (24.8939, 67.0656),  # Karachi
    "Multan Cricket Stadium": (30.1971, 71.4407),
    "Rawalpindi Cricket Stadium": (33.6440, 73.0822),
    "Iqbal Stadium": (31.4267, 73.0944),  # Faisalabad
    "Kensington Oval": (13.1078, -59.6240),  # Bridgetown
    "Daren Sammy National Cricket Stadium": (14.0167, -60.9500),  # Gros Islet
    "Queen's Park Oval": (10.6553, -61.5081),  # Port of Spain
    "Brian Lara Stadium": (10.2917, -61.4575),  # Tarouba
    "Sabina Park": (17.9722, -76.7720),  # Kingston
    "Providence Stadium": (6.7822, -58.2095),  # Guyana
    "Warner Park": (17.3000, -62.7167),  # Basseterre
    "National Cricket Stadium": (12.0444, -61.7372),  # St George's
    "Hagley Oval": (-43.5328, 172.6309),  # Christchurch
    "Eden Park": (-36.8744, 174.7458),  # Auckland
    "Bay Oval": (-37.6611, 176.1819),  # Mount Maunganui
    "Basin Reserve": (-41.2978, 174.7795),  # Wellington
    "Westpac Stadium": (-41.2733, 174.7867),  # Wellington
    "Seddon Park": (-37.7833, 175.2833),  # Hamilton
    "University Oval": (-45.8612, 170.5169),  # Dunedin
}


@dataclass(frozen=True)
class WeatherReading:
    venue: str
    coord_lat: float
    coord_lon: float
    target_iso: str
    temp_c: float | None
    humidity_pct: float | None
    wind_kph: float | None
    precip_mm: float | None
    cloud_pct: float | None
    source: str  # 'archive' | 'forecast' | 'cached'

    def summary(self) -> str:
        bits = []
        if self.temp_c is not None:
            bits.append(f"{self.temp_c:.0f}°C")
        if self.humidity_pct is not None:
            bits.append(f"hum {self.humidity_pct:.0f}%")
        if self.wind_kph is not None:
            bits.append(f"wind {self.wind_kph:.0f}kph")
        if self.precip_mm is not None and self.precip_mm > 0:
            bits.append(f"rain {self.precip_mm:.1f}mm")
        if self.cloud_pct is not None:
            bits.append(f"cloud {self.cloud_pct:.0f}%")
        return ", ".join(bits) if bits else "no data"

    def is_hot_dry(self) -> bool:
        """Heuristic: hot + low humidity + no recent rain = batting-friendly."""
        if self.temp_c is None: return False
        return self.temp_c >= 25 and (self.humidity_pct or 100) < 65 and (self.precip_mm or 0) < 1


def _strip_city_suffix(venue: str) -> str:
    """Normalize 'M Chinnaswamy Stadium, Bengaluru' -> 'M Chinnaswamy Stadium'."""
    return venue.split(",")[0].strip()


def _load_cache() -> dict:
    if VENUE_COORDS_FILE.exists():
        return json.loads(VENUE_COORDS_FILE.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    VENUE_COORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    VENUE_COORDS_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def geocode_venue(venue: str, *, allow_network: bool = True) -> tuple[float, float] | None:
    """Look up (lat, lon) for a venue name. Order:
       1. exact match in cache
       2. exact match in VENUE_OVERRIDES
       3. stem-match in overrides (drop ', City')
       4. Open-Meteo geocoding API call (if allow_network)
    """
    cache = _load_cache()
    if venue in cache:
        c = cache[venue]
        return (c["lat"], c["lon"]) if c.get("lat") is not None else None
    if venue in VENUE_OVERRIDES:
        return VENUE_OVERRIDES[venue]
    stem = _strip_city_suffix(venue)
    if stem in VENUE_OVERRIDES:
        return VENUE_OVERRIDES[stem]
    if not allow_network:
        return None
    # Geocoding API call
    try:
        q = urlencode({"name": stem, "count": 1, "language": "en", "format": "json"})
        url = f"https://geocoding-api.open-meteo.com/v1/search?{q}"
        req = Request(url, headers={"User-Agent": "cricket-trading/1.0"})
        with urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        results = data.get("results") or []
        if not results:
            cache[venue] = {"lat": None, "lon": None, "note": "geocoding_no_match"}
            _save_cache(cache)
            return None
        first = results[0]
        lat, lon = float(first["latitude"]), float(first["longitude"])
        cache[venue] = {"lat": lat, "lon": lon, "name": first.get("name"), "country": first.get("country")}
        _save_cache(cache)
        time.sleep(0.5)  # be polite
        return (lat, lon)
    except Exception as e:
        logger.warning("geocode failed for %r: %s", venue, e)
        return None


def fetch_weather(venue: str, target: date | datetime, *, hour_local: int = 18) -> WeatherReading | None:
    """Fetch weather for `venue` at `target` (date or datetime). For dates,
    use `hour_local` (default 18:00 = typical T20 evening start) as the lookup hour."""
    coord = geocode_venue(venue)
    if coord is None:
        return None
    lat, lon = coord

    if isinstance(target, datetime):
        target_date = target.date()
        target_hour = target.hour
    else:
        target_date = target
        target_hour = hour_local

    today = date.today()
    is_historical = target_date < today

    base = "https://archive-api.open-meteo.com/v1/archive" if is_historical \
        else "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,cloud_cover",
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }
    if is_historical:
        params["start_date"] = target_date.isoformat()
        params["end_date"] = target_date.isoformat()
    else:
        params["start_date"] = target_date.isoformat()
        params["end_date"] = target_date.isoformat()

    url = f"{base}?{urlencode(params)}"
    try:
        req = Request(url, headers={"User-Agent": "cricket-trading/1.0"})
        with urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        logger.warning("weather fetch failed for %r %s: %s", venue, target_date, e)
        return None

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None
    # Find the hourly slot closest to target_hour
    target_iso = f"{target_date.isoformat()}T{target_hour:02d}:00"
    idx = None
    for i, t in enumerate(times):
        if t.startswith(target_iso[:13]):  # match YYYY-MM-DDTHH
            idx = i
            break
    if idx is None:
        # fallback to mid-day
        idx = min(len(times) - 1, 18) if len(times) > 18 else len(times) // 2
    def get(k):
        arr = hourly.get(k) or []
        return float(arr[idx]) if idx < len(arr) and arr[idx] is not None else None
    return WeatherReading(
        venue=venue, coord_lat=lat, coord_lon=lon, target_iso=times[idx],
        temp_c=get("temperature_2m"),
        humidity_pct=get("relative_humidity_2m"),
        wind_kph=get("wind_speed_10m"),
        precip_mm=get("precipitation"),
        cloud_pct=get("cloud_cover"),
        source="archive" if is_historical else "forecast",
    )


if __name__ == "__main__":
    # Quick smoke test
    from src.logging_setup import configure_logging
    configure_logging()

    test_cases = [
        ("The Rose Bowl", date(2026, 5, 28), 18),  # Today, Hampshire v Essex
        ("M Chinnaswamy Stadium", date(2026, 4, 1), 20),  # IPL
        ("Sophia Gardens", date(2026, 5, 22), 18),  # NTB
        ("Lord's", date(2026, 6, 15), 11),  # Test cricket morning start
    ]
    for venue, d, hr in test_cases:
        w = fetch_weather(venue, d, hour_local=hr)
        print(f"{venue} @ {d.isoformat()} {hr}:00 -> {w.summary() if w else 'NO DATA'}")
