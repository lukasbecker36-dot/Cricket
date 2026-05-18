"""Identify IPL Match Odds markets inside a Betfair BASIC tar archive.

Strategy: stream each .bz2 entry, read just the first JSON record, parse the
marketDefinition, and check if eventName contains an IPL team name. We do NOT
extract files to disk during scanning.
"""
from __future__ import annotations

import bz2
import json
import logging
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Current IPL franchise names plus former names that appear in older seasons.
IPL_TEAMS: frozenset[str] = frozenset({
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Royal Challengers Bengaluru",  # rebrand for 2024 onward
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Delhi Daredevils",  # pre-2019 name
    "Sunrisers Hyderabad",
    "Punjab Kings",
    "Kings XI Punjab",  # pre-2021 name
    "Rajasthan Royals",
    "Lucknow Super Giants",
    "Gujarat Titans",
    "Rising Pune Supergiant",  # 2016-2017
    "Rising Pune Supergiants",
    "Gujarat Lions",  # 2016-2017
    "Pune Warriors",  # 2011-2013
    "Pune Warriors India",
    "Kochi Tuskers Kerala",  # 2011
    "Deccan Chargers",  # 2008-2012
})


def is_ipl_event(event_name: str | None, runners: list[dict] | None) -> bool:
    """An IPL match has both runner teams in our IPL set."""
    if runners is None:
        return False
    runner_names = {r.get("name", "") for r in runners}
    if len(runner_names) < 2:
        return False
    return runner_names.issubset(IPL_TEAMS) or all(n in IPL_TEAMS for n in runner_names if n)


def first_market_definition(bz2_bytes: bytes) -> dict | None:
    """Decompress the first record only; cheap header read."""
    try:
        decompressor = bz2.BZ2Decompressor()
        # 64 KB is enough for the first line in practice
        head = decompressor.decompress(bz2_bytes[:65536])
        first_newline = head.find(b"\n")
        if first_newline == -1:
            return None
        first_line = head[:first_newline].decode("utf-8")
        obj = json.loads(first_line)
        for change in obj.get("mc", []):
            if "marketDefinition" in change:
                return change["marketDefinition"]
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("could not parse header: %s", e)
    return None


def scan_archive(tar_path: Path) -> list[dict]:
    """Yield one record per market found inside the archive."""
    results: list[dict] = []
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".bz2"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            md = first_market_definition(f.read())
            if md is None:
                continue
            event_name = md.get("eventName")
            runners = md.get("runners", [])
            if not is_ipl_event(event_name, runners):
                continue
            results.append({
                "archive_path": member.name,
                "event_id": md.get("eventId"),
                "market_id": md.get("name") and member.name.rsplit("/", 1)[-1].replace(".bz2", ""),
                "market_type": md.get("marketType"),
                "market_time": md.get("marketTime"),
                "event_name": event_name,
                "runners": [(r.get("id"), r.get("name")) for r in runners],
            })
    return results


def main() -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", default=Path("data/processed/betfair_ipl_index.json"), type=Path)
    args = parser.parse_args()

    matches = scan_archive(args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(matches, indent=2))
    logger.info("found %d IPL markets; index written to %s", len(matches), args.out)

    # Quick season breakdown
    by_season: dict[str, int] = {}
    for m in matches:
        season = (m.get("market_time") or "")[:4]
        if season:
            by_season[season] = by_season.get(season, 0) + 1
    for s in sorted(by_season):
        print(f"  {s}: {by_season[s]} markets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
