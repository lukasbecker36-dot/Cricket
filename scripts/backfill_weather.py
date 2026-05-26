"""Backfill historical weather for every match in our training data.

Iterates over each unique (venue, year), fetches one year of hourly weather
from Open-Meteo archive API, extracts the 18:00 local reading for each match
date at that venue. Saves to data/processed/match_weather.parquet.

Resumable: if partial output exists, skips venue-years already saved.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request

import pandas as pd

from src.live.weather import geocode_venue
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]
OUTPUT = Path("data/processed/match_weather.parquet")
MATCH_HOUR_LOCAL = 18  # T20s mostly evening starts; we extract 18:00 local reading


def collect_pairs() -> dict:
    """Return {venue: set(date_str)} across all leagues."""
    pairs: dict[str, set] = defaultdict(set)
    for league in LEAGUES:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        m = pd.read_parquet(d / "matches.parquet")
        for _, r in m.iterrows():
            v = r.get("venue")
            d_str = str(r.get("date"))[:10]
            if v and d_str and d_str not in ("nan", "NaT"):
                pairs[v].add(d_str)
    return pairs


def fetch_year_archive(lat: float, lon: float, year: int) -> dict | None:
    """Fetch one year of hourly weather for (lat, lon)."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date":   f"{year}-12-31",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,cloud_cover",
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }
    url = f"https://archive-api.open-meteo.com/v1/archive?{urlencode(params)}"
    try:
        req = Request(url, headers={"User-Agent": "cricket-trading/1.0"})
        with urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        logger.warning("archive fetch failed lat=%s lon=%s year=%s: %s", lat, lon, year, e)
        return None


def extract_for_date(archive: dict, target_date: str, target_hour: int) -> dict | None:
    hourly = archive.get("hourly") or {}
    times = hourly.get("time") or []
    if not times: return None
    needle = f"{target_date}T{target_hour:02d}:00"
    for i, t in enumerate(times):
        if t == needle:
            def get(k):
                arr = hourly.get(k) or []
                return float(arr[i]) if i < len(arr) and arr[i] is not None else None
            return {
                "temp_c": get("temperature_2m"),
                "humidity_pct": get("relative_humidity_2m"),
                "wind_kph": get("wind_speed_10m"),
                "precip_mm": get("precipitation"),
                "cloud_pct": get("cloud_cover"),
            }
    return None


def main() -> int:
    configure_logging()
    pairs = collect_pairs()
    logger.info("Unique venues: %d, total (venue,date) pairs: %d",
                len(pairs), sum(len(v) for v in pairs.values()))

    # Resume support: load existing rows
    existing = pd.read_parquet(OUTPUT) if OUTPUT.exists() else pd.DataFrame(
        columns=["venue", "date", "temp_c", "humidity_pct", "wind_kph", "precip_mm", "cloud_pct"]
    )
    done = set(zip(existing["venue"], existing["date"]))
    logger.info("Already have %d rows", len(existing))

    new_rows = []
    n_skipped_venue = 0
    n_api_calls = 0

    for venue, dates in pairs.items():
        # Skip if all dates for this venue already done
        remaining = sorted(d for d in dates if (venue, d) not in done)
        if not remaining:
            continue

        coord = geocode_venue(venue)
        if coord is None:
            logger.warning("no coords for %r, skipping %d dates", venue, len(remaining))
            n_skipped_venue += 1
            continue
        lat, lon = coord

        # Group by year
        by_year: dict[int, list] = defaultdict(list)
        for d in remaining:
            by_year[int(d[:4])].append(d)

        for year, year_dates in sorted(by_year.items()):
            archive = fetch_year_archive(lat, lon, year)
            n_api_calls += 1
            if archive is None:
                continue
            for d in year_dates:
                row = extract_for_date(archive, d, MATCH_HOUR_LOCAL)
                if row is None:
                    continue
                row.update({"venue": venue, "date": d})
                new_rows.append(row)
            time.sleep(0.2)  # be polite

        # Periodic flush
        if len(new_rows) >= 200:
            combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            combined.to_parquet(OUTPUT, index=False)
            existing = combined
            done = set(zip(existing["venue"], existing["date"]))
            logger.info("Flushed: %d total rows, %d API calls", len(existing), n_api_calls)
            new_rows = []

    if new_rows:
        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        combined.to_parquet(OUTPUT, index=False)
        logger.info("Final flush: %d total rows", len(combined))

    logger.info("Done. API calls=%d  venues_skipped_no_coords=%d", n_api_calls, n_skipped_venue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
